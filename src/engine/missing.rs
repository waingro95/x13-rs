//! `editor.f` missing-value placeholders, kept separate from ordinary observations.
//!
//! A NaN observation is a gap. Build 62 marks gaps with its `-99999.0` code
//! (gtinpt.f's `Mvcode`); this crate has no such code, so `-99999.0` is an
//! ordinary observation (a deliberate difference from Build 62).
use std::borrow::Cow;

use crate::time::TimeAxis;
use crate::{Error, Frequency, Period};

pub(crate) const PLACEHOLDER: f64 = 1_000_000_000.0;

pub(crate) struct PreparedInput<'a> {
    pub(crate) values: Cow<'a, [f64]>,
    pub(crate) missing: Vec<usize>,
}

impl<'a> PreparedInput<'a> {
    pub(crate) fn new(values: &'a [f64], axis: TimeAxis) -> Result<Self, Error> {
        let missing = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_nan().then_some(index))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(Self {
                values: Cow::Borrowed(values),
                missing,
            });
        }
        let mut seasons = SeasonCoverage::new(axis.frequency());
        let mut prepared = values.to_vec();
        let filler = gap_filler(values);
        for (index, period) in axis.periods(values.len())?.enumerate() {
            if missing.binary_search(&index).is_ok() {
                // editor.f substitutes before transformation.
                prepared[index] = filler;
            } else {
                seasons.observe(period);
            }
        }
        seasons.require_all()?;
        Ok(Self {
            values: Cow::Owned(prepared),
            missing,
        })
    }
}

/// What fills the gaps: gtinpt.f's `Mvval`, except in a constant series.
///
/// Build 62 skips the model on a constant series, so nothing estimates its gaps
/// and the 1e9 filler reaches X-11 as data, giving values the series cannot
/// produce. The constant fills them instead, which is what Build 62 does when
/// `missingval` is set to it.
fn gap_filler(values: &[f64]) -> f64 {
    if crate::numerics::is_constant_with_gaps(values) {
        values[0]
    } else {
        PLACEHOLDER
    }
}

/// x11ari.f's `IF(Same)` on `values`, which hold the filler at `missing`.
///
/// A leading gap never counts: editor.f seeds its extremes from the missing
/// code, so the model runs on that series even if filling leaves it constant.
pub(super) fn skips_model(values: &[f64], missing: &[usize]) -> bool {
    missing.first() != Some(&0) && crate::numerics::is_constant_series(values)
}

/// editor.f's present-season check: every month or quarter of the year needs
/// at least one observation.
pub(super) struct SeasonCoverage {
    frequency: Frequency,
    present: [bool; 12],
}

impl SeasonCoverage {
    pub(super) fn new(frequency: Frequency) -> Self {
        Self {
            frequency,
            present: [false; 12],
        }
    }

    pub(super) fn observe(&mut self, period: Period) {
        self.present[usize::from(period.subperiod() - 1)] = true;
    }

    /// `MissingSeason` for the first month or quarter never observed.
    pub(super) fn require_all(&self) -> Result<(), Error> {
        let seasons = usize::from(self.frequency.periods_per_year());
        match self.present[..seasons].iter().position(|&present| !present) {
            Some(index) => Err(Error::MissingSeason {
                frequency: self.frequency,
                subperiod: (index + 1) as u8,
            }),
            None => Ok(()),
        }
    }
}

/// x11pt1.f resets missing placeholders after removing observed prior factors.
/// Only a fitted model gets here, so the filler is always [`PLACEHOLDER`].
pub(super) fn reset(values: &mut [f64], missing: &[usize]) {
    for &index in missing {
        values[index] = PLACEHOLDER;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_quarter_needs_an_observation() {
        let axis = TimeAxis::quarterly(2000, 2).unwrap();
        let level = |index: usize| 100.0 + index as f64;
        let mut values = (0..12).map(level).collect::<Vec<_>>();
        // The third quarters sit at 1, 5 and 9; the first quarters at 3, 7, 11.
        for index in [1, 3, 5, 7, 9, 11] {
            values[index] = f64::NAN;
        }
        let missing_season = |subperiod| Error::MissingSeason {
            frequency: crate::Frequency::Quarterly,
            subperiod,
        };
        assert_eq!(
            PreparedInput::new(&values, axis).err(),
            Some(missing_season(1))
        );
        values[7] = level(7);
        assert_eq!(
            PreparedInput::new(&values, axis).err(),
            Some(missing_season(3))
        );
        values[5] = level(5);
        let prepared = PreparedInput::new(&values, axis).unwrap();
        assert_eq!(prepared.missing, [1, 3, 9, 11]);
        for index in 0..values.len() {
            let expected = if prepared.missing.contains(&index) {
                PLACEHOLDER
            } else {
                level(index)
            };
            assert_eq!(prepared.values[index].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn only_a_constant_series_fills_with_its_constant() {
        let filled = |values: &[f64]| {
            let axis = TimeAxis::quarterly(2000, 1).unwrap();
            PreparedInput::new(values, axis).unwrap().values[1]
        };
        let series = |first, last| {
            let mut values = vec![5.0; 8];
            values[1] = f64::NAN;
            values[0] = first;
            values[7] = last;
            values
        };
        assert_eq!(filled(&series(5.0, 5.0)), 5.0);
        assert_eq!(filled(&series(5.0, 6.0)), PLACEHOLDER);
        // A leading gap is never `Same`, so the model estimates both gaps.
        assert_eq!(filled(&series(f64::NAN, 5.0)), PLACEHOLDER);
    }

    /// `dpeq.f` calls two values equal when they differ by less than
    /// 3.834e-20, whatever their size.
    #[test]
    fn a_series_is_constant_below_the_dpeq_threshold() {
        let threshold: f64 = 3.834e-20;
        let below = f64::from_bits(threshold.to_bits() - 1);
        assert!(skips_model(&[0.0, below], &[]));
        assert!(!skips_model(&[0.0, threshold], &[]));
        assert!(!skips_model(&[1e20, 1e20 + 1e6], &[]));
        assert!(!skips_model(&[5.0, 5.0], &[0]));
    }
}
