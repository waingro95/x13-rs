//! The fitted regression effects around a SEATS run.
//!
//! ADJREG (`adjreg.f`) removes the regression effects from the series before
//! SEATS decomposes it, and DETCOMP (`ansub4.f`) puts them back into the
//! final components. [`FittedEffectsInput`] holds what ADJREG leaves: the
//! series the extraction reads, TRAM (the series DETCOMP adjusts, with only
//! the missing-value effects removed) and each effect in the form DETCOMP
//! applies it: a factor under the log transform, a term otherwise.
//! [`final_with_effects`] and its variants are DETCOMP's final components.

use super::airline::SeatsTransform;
use super::extraction::{
    ExtractedComponents, FinalComponents, FinalWithTransitory, SAVED_IRREGULAR, SAVED_TREND,
};
use super::failure::{Error, Failure, Stage};
use super::trading_day::EffectiveTradingDay;
use crate::time::TimeAxis;
use SeatsEffectSeries as Series;

/// A series of the fitted effects, or a value made from them, as a failure
/// names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeatsEffectSeries {
    Original,
    Transformed,
    LevelShiftLinear,
    AdditiveOutlierLinear,
    TemporaryChangeLinear,
    MissingValueLinear,
    TradingDayLinear,
    HolidayLinear,
    UserLinear,
    FixedLength,
    Oz,
    Tram,
    Patd,
    Paeast,
    Paoutr,
    Paouir,
    LoggedOz,
    SeasonallyAdjustedDivisor,
    SeasonallyAdjusted,
}

impl SeatsEffectSeries {
    /// The name of the series in a failure.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Original => "original series",
            Self::Transformed => "transformed series",
            Self::LevelShiftLinear => "level-shift effect",
            Self::AdditiveOutlierLinear => "additive-outlier effect",
            Self::TemporaryChangeLinear => "temporary-change effect",
            Self::MissingValueLinear => "missing-value effect",
            Self::TradingDayLinear => "trading-day effect",
            Self::HolidayLinear => "holiday effect",
            Self::UserLinear => "user-defined regression effect",
            Self::FixedLength => "fixed length-of-period effect",
            Self::Oz => "SEATS input series",
            Self::Tram => "linearized series",
            Self::Patd => "SEATS trading-day effect",
            Self::Paeast => "SEATS Easter effect",
            Self::Paoutr => "SEATS trend outlier effect",
            Self::Paouir => "SEATS irregular outlier effect",
            Self::LoggedOz => "logged SEATS input series",
            Self::SeasonallyAdjustedDivisor => "seasonally adjusted divisor",
            Self::SeasonallyAdjusted => "seasonally adjusted series",
        }
    }
}

/// ADJREG's results for one series, each vector aligned with it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FittedEffectsInput {
    axis: TimeAxis,
    transform: SeatsTransform,
    original: Vec<f64>,
    transformed: Vec<f64>,
    /// The series less its missing-value effects, on the original scale.
    tram: Vec<f64>,
    /// The series less every effect, on the transformed scale: what the
    /// passes and the extraction read.
    extraction_response: Vec<f64>,
    /// The trading-day effect, with the fixed-length factor under the log.
    patd: Vec<f64>,
    /// The holiday effect.
    paeast: Vec<f64>,
    /// The level shifts, which DETCOMP gives the trend.
    paoutr: Vec<f64>,
    /// The additive outliers and temporary changes, which DETCOMP gives the
    /// irregular.
    paouir: Vec<f64>,
    requires_final_adjustment: bool,
}

/// ADJREG's regression groups other than the outliers, each aligned with the
/// series. Under the log transform `fixed_length` is a factor and must be
/// positive; the others are linear effects.
#[derive(Clone, Copy)]
pub(crate) struct RegressionEffects<'a> {
    pub(crate) trading_day: &'a [f64],
    pub(crate) holiday: &'a [f64],
    pub(crate) fixed_length: &'a [f64],
    pub(crate) missing: &'a [f64],
    pub(crate) user: &'a [f64],
}

/// The outlier effects on the transformed scale, each aligned with the
/// series.
pub(crate) struct OutlierLinearEffects<'a> {
    /// Whether the model has a trend or irregular outlier, so that the
    /// seasonally adjusted series of a model without seasonal roots is not
    /// TRAM.
    pub(crate) requires_final_adjustment: bool,
    pub(crate) level_shift: &'a [f64],
    pub(crate) additive: &'a [f64],
    pub(crate) temporary_change: &'a [f64],
}

/// Whether ADJREG multiplies TRAM by the fixed-length factor (`Kfmt > 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeriesPriorRestoration {
    Restore,
    Omit,
}

/// The outlier effects DETCOMP adds to the final trend and irregular.
#[derive(Clone, Copy)]
pub(super) struct FinalOutlierEffects<'a> {
    pub(super) trend: &'a [f64],
    pub(super) irregular: &'a [f64],
}

impl FittedEffectsInput {
    pub(crate) fn requires_final_adjustment(&self) -> bool {
        self.requires_final_adjustment
    }

    /// ADJREG for a model without missing values, user regressors or
    /// outliers.
    pub(crate) fn new(
        axis: TimeAxis,
        transform: SeatsTransform,
        original: &[f64],
        transformed: &[f64],
        trading_day_linear: &[f64],
        holiday_linear: &[f64],
        fixed_length: &[f64],
    ) -> Result<Self, Error> {
        let neutral = vec![0.0; original.len()];
        Self::with_outliers(
            axis,
            transform,
            original,
            transformed,
            RegressionEffects {
                trading_day: trading_day_linear,
                holiday: holiday_linear,
                fixed_length,
                missing: &neutral,
                user: &neutral,
            },
            SeriesPriorRestoration::Restore,
            OutlierLinearEffects {
                requires_final_adjustment: false,
                level_shift: &neutral,
                additive: &neutral,
                temporary_change: &neutral,
            },
        )
    }

    /// ADJREG: checks every input and removes the effects from the series.
    pub(crate) fn with_outliers(
        axis: TimeAxis,
        transform: SeatsTransform,
        original: &[f64],
        transformed: &[f64],
        regression: RegressionEffects<'_>,
        prior_restoration: SeriesPriorRestoration,
        outliers: OutlierLinearEffects<'_>,
    ) -> Result<Self, Error> {
        let RegressionEffects {
            trading_day: trading_day_linear,
            holiday: holiday_linear,
            fixed_length,
            missing,
            user,
        } = regression;
        let n = original.len();
        debug_assert!(
            [
                transformed,
                trading_day_linear,
                holiday_linear,
                fixed_length,
                missing,
                user,
                outliers.level_shift,
                outliers.additive,
                outliers.temporary_change,
            ]
            .iter()
            .all(|values| values.len() == n)
        );
        let check = |series, index, value, positive| {
            checked(
                Stage::FittedEffectsPreparation,
                series,
                index,
                value,
                positive,
            )
        };
        let mut result = Self {
            axis,
            transform,
            original: original.to_vec(),
            transformed: transformed.to_vec(),
            tram: Vec::with_capacity(n),
            extraction_response: Vec::with_capacity(n),
            patd: Vec::with_capacity(n),
            paeast: Vec::with_capacity(n),
            paoutr: Vec::with_capacity(n),
            paouir: Vec::with_capacity(n),
            requires_final_adjustment: outliers.requires_final_adjustment,
        };
        let log = transform == SeatsTransform::Log;
        // The ten groups are parallel, equal-length slices, so one index
        // walks them in ADJREG's per-observation check order.
        for i in 0..n {
            for (series, value, positive) in [
                (Series::Original, original[i], log),
                (Series::Transformed, transformed[i], false),
                (Series::TradingDayLinear, trading_day_linear[i], false),
                (Series::HolidayLinear, holiday_linear[i], false),
                (Series::FixedLength, fixed_length[i], log),
                (Series::MissingValueLinear, missing[i], false),
                (Series::UserLinear, user[i], false),
                (Series::LevelShiftLinear, outliers.level_shift[i], false),
                (Series::AdditiveOutlierLinear, outliers.additive[i], false),
                (
                    Series::TemporaryChangeLinear,
                    outliers.temporary_change[i],
                    false,
                ),
            ] {
                check(series, i, value, positive)?;
            }
            // adjreg:50–55: keep all ten operations, including neutral groups.
            let adjusted = [
                trading_day_linear[i],
                outliers.level_shift[i],
                holiday_linear[i],
                outliers.additive[i],
                outliers.temporary_change[i],
                user[i],
                missing[i],
                0.0,
                0.0,
                0.0,
            ]
            .into_iter()
            .try_fold(transformed[i], |adjusted, effect| {
                check(Series::Oz, i, adjusted - effect, false)
            })?;
            let orixmv = check(Series::Tram, i, transformed[i] - missing[i], false)?;
            let (tram, extraction, patd, paeast) = if log {
                let oz = check(Series::Oz, i, adjusted.exp(), true)?;
                let tram = check(Series::Tram, i, orixmv.exp(), true)?;
                // adjreg restores Sprior to Orixmv only when Kfmt > 0.
                // x11pt2/tdlom still folds it into Factd independently.
                let tram = if prior_restoration == SeriesPriorRestoration::Restore {
                    check(Series::Tram, i, tram * fixed_length[i], true)?
                } else {
                    tram
                };
                let extraction = check(Series::LoggedOz, i, oz.ln(), false)?;
                let patd = check(Series::Patd, i, trading_day_linear[i].exp(), true)?;
                let patd = check(Series::Patd, i, patd * fixed_length[i], true)?;
                let patd = check(Series::Patd, i, patd / 1.0, true)?;
                let paeast = check(Series::Paeast, i, holiday_linear[i].exp(), true)?;
                (tram, extraction, patd, paeast)
            } else {
                (orixmv, adjusted, trading_day_linear[i], holiday_linear[i])
            };
            result.tram.push(tram);
            result.extraction_response.push(extraction);
            result.patd.push(patd);
            result.paeast.push(paeast);
            let (paoutr, paouir) = if log {
                let paoutr = check(Series::Paoutr, i, outliers.level_shift[i].exp(), true)?;
                let ao = check(Series::Paouir, i, outliers.additive[i].exp(), true)?;
                let tc = check(Series::Paouir, i, outliers.temporary_change[i].exp(), true)?;
                (paoutr, check(Series::Paouir, i, ao * tc, true)?)
            } else {
                (
                    outliers.level_shift[i],
                    check(
                        Series::Paouir,
                        i,
                        outliers.additive[i] + outliers.temporary_change[i],
                        false,
                    )?,
                )
            };
            result.paoutr.push(paoutr);
            result.paouir.push(paouir);
        }
        Ok(result)
    }
    pub(crate) fn axis(&self) -> TimeAxis {
        self.axis
    }
    pub(crate) fn tram(&self) -> &[f64] {
        &self.tram
    }
    pub(crate) fn extraction_response(&self) -> &[f64] {
        &self.extraction_response
    }
    pub(crate) fn patd(&self) -> &[f64] {
        &self.patd
    }
    pub(crate) fn paeast(&self) -> &[f64] {
        &self.paeast
    }
    pub(crate) fn paoutr(&self) -> &[f64] {
        &self.paoutr
    }
    pub(crate) fn paouir(&self) -> &[f64] {
        &self.paouir
    }
}

/// `value`, or [`Failure::Unusable`] naming `series` at `index`, raised at
/// `stage`, when it is not finite or, where `positive` is set, not above zero.
pub(crate) fn checked(
    stage: Stage,
    series: Series,
    index: usize,
    value: f64,
    positive: bool,
) -> Result<f64, Error> {
    if value.is_finite() && (!positive || value > 0.0) {
        Ok(value)
    } else {
        Err(Failure::Unusable {
            quantity: series.label(),
            index: Some(index),
            value,
        }
        .at(stage))
    }
}

/// DETCOMP's final components of a model without a transitory component.
pub(crate) fn final_with_effects(
    input: &FittedEffectsInput,
    raw: &ExtractedComponents,
) -> Result<FinalComponents, Error> {
    debug_assert!(raw.cycle.is_none());
    Ok(final_with_effects_parts(input, raw, EffectiveTradingDay::Ordinary)?.0)
}

/// DETCOMP's final components and the saved transitory component.
pub(super) fn final_with_effects_and_transitory(
    input: &FittedEffectsInput,
    raw: &ExtractedComponents,
    cycle_variance: f64,
) -> Result<FinalWithTransitory, Error> {
    final_with_effects_ar2(input, raw, cycle_variance, EffectiveTradingDay::Ordinary)
}

/// [`final_with_effects_and_transitory`] for a model whose transitory
/// component SIGEX may treat as a stochastic trading day.
pub(super) fn final_with_effects_ar2(
    input: &FittedEffectsInput,
    raw: &ExtractedComponents,
    cycle_variance: f64,
    effective: EffectiveTradingDay,
) -> Result<FinalWithTransitory, Error> {
    if input.transform == SeatsTransform::Log {
        super::log_transform::finite(cycle_variance, "saved cycle variance", None)?;
    }
    let (saved, working_cycle) = final_with_effects_parts(input, raw, effective)?;
    let transitory = match input.transform {
        SeatsTransform::Identity => {
            super::extraction::final_identity_transitory(raw, cycle_variance)?
        }
        SeatsTransform::Log => {
            super::log_transform::final_log_transitory(working_cycle, cycle_variance)
        }
    };
    Ok(FinalWithTransitory { saved, transitory })
}

/// DETCOMP's saved components with the ADJREG effects restored, plus the log
/// model's working cycle.
fn final_with_effects_parts(
    input: &FittedEffectsInput,
    raw: &ExtractedComponents,
    effective: EffectiveTradingDay,
) -> Result<(FinalComponents, Option<Vec<f64>>), Error> {
    debug_assert_eq!(raw.frequency, input.axis.frequency());
    debug_assert_eq!(raw.observed_len, input.original.len());
    debug_assert!(
        !(effective == EffectiveTradingDay::Stochastic
            && input.transform == SeatsTransform::Identity
            && raw
                .cycle
                .as_ref()
                .is_some_and(|cycle| cycle.len() != raw.observed_len))
    );
    let check = |series, i, value, positive| {
        checked(Stage::FittedEffectsRestoration, series, i, value, positive)
    };
    let effects = Some(FinalOutlierEffects {
        trend: input.paoutr(),
        irregular: input.paouir(),
    });
    let restored = match input.transform {
        SeatsTransform::Identity => {
            super::extraction::final_identity_with(input.tram(), raw, effects, |i, sc| {
                let divisor = [input.paeast()[i], 0.0, input.patd()[i], 0.0, 0.0]
                    .into_iter()
                    .try_fold(sc, |divisor, effect| {
                        check(
                            Series::SeasonallyAdjustedDivisor,
                            i,
                            divisor + effect,
                            false,
                        )
                    })?;
                let sa = check(
                    Series::SeasonallyAdjusted,
                    i,
                    input.tram()[i] - divisor,
                    false,
                )?;
                if effective == EffectiveTradingDay::Ordinary {
                    return Ok::<_, Error>(sa);
                }
                // DETCOMP ansub4:1987–1992: subtract cycle+Pareg5 after the
                // ordinary calendar divisor, not within it.
                let cycle = check(
                    Series::SeasonallyAdjustedDivisor,
                    i,
                    raw.cycle.as_ref().map_or(0.0, |c| c[i]) + 0.0,
                    false,
                )?;
                check(Series::SeasonallyAdjusted, i, sa - cycle, false)
            })
            .map(|saved| (saved, None))
        }
        SeatsTransform::Log => {
            let log = super::log_transform::LogInput::from_effects(input);
            super::log_transform::final_log_with(
                &log,
                raw,
                effects,
                effective,
                |i, sc, cycle_percent| {
                    // DETCOMP resets bias1 to one; SECOND's biases remain upstream.
                    let bias1 = 1.0;
                    let divisor = check(Series::SeasonallyAdjustedDivisor, i, sc / 100.0, true)?;
                    let divisor = [input.paeast()[i], 1.0, input.patd()[i], 1.0, bias1, 1.0]
                        .into_iter()
                        .try_fold(divisor, |divisor, effect| {
                            check(Series::SeasonallyAdjustedDivisor, i, divisor * effect, true)
                        })?;
                    let sa = check(
                        Series::SeasonallyAdjusted,
                        i,
                        input.tram()[i] / divisor,
                        true,
                    )?;
                    if effective == EffectiveTradingDay::Ordinary {
                        return Ok::<_, Error>(sa);
                    }
                    // DETCOMP ansub4:2855–2861; working SECOND cycle,
                    // independent of S14 availability. Pareg5 is one.
                    let cycle = check(
                        Series::SeasonallyAdjustedDivisor,
                        i,
                        1.0 * cycle_percent,
                        true,
                    )?;
                    let cycle = check(Series::SeasonallyAdjustedDivisor, i, cycle / 100.0, true)?;
                    check(Series::SeasonallyAdjusted, i, sa / cycle, true)
                },
            )
        }
    };
    restored.map_err(saved_component_failure)
}

/// Restages an unusable saved trend or irregular, where DETCOMP applies the
/// outlier effects, to the fitted-effects restoration. Every other failure
/// keeps its stage.
fn saved_component_failure(error: Error) -> Error {
    if matches!(
        error.failure,
        Failure::Unusable {
            quantity: SAVED_TREND | SAVED_IRREGULAR,
            index: Some(_),
            ..
        }
    ) {
        error.restage(Stage::FittedEffectsRestoration)
    } else {
        error
    }
}

#[cfg(test)]
mod tests;
