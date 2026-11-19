//! The log transformation around a SEATS run.
//!
//! Under a log transformation the X-13 driver exponentiates the transformed
//! series before it calls SEATS, and ANALTS takes the log again, so SEATS
//! decomposes `ln(exp(x))` and not `x`. [`LogInput`] holds that series.
//!
//! [`final_log_with`] is SECOND (`sigsub.f`): it corrects the bias of the
//! exponentiated components and converts them to the original scale. The other
//! `final_log*` functions add what DETCOMP and USRENTRY do to the result.

use super::extraction::{ExtractedComponents, FinalComponents, SAVED_IRREGULAR, SAVED_TREND};
use super::failure::{Error, Failure, Stage, usable};
use super::residual::MAXIMUM_OBSERVATIONS;
use super::trading_day::EffectiveTradingDay;
use crate::Frequency;

/// The series of a log run, as SEATS receives them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LogInput {
    frequency: Frequency,
    /// Tram, the linearized series on the original scale.
    tram: Vec<f64>,
    /// The log ANALTS takes of the series it receives.
    logged: Vec<f64>,
}

impl LogInput {
    pub(super) fn from_effects(input: &super::effects::FittedEffectsInput) -> Self {
        Self {
            frequency: input.axis().frequency(),
            tram: input.tram().to_vec(),
            logged: input.extraction_response().to_vec(),
        }
    }
    pub(crate) fn logged(&self) -> &[f64] {
        &self.logged
    }
}

/// The log input of a run without fitted effects, from the original series
/// and its log.
pub(super) fn prepare_transformed_log_input(
    frequency: Frequency,
    original: &[f64],
    transformed: &[f64],
) -> Result<LogInput, Error> {
    let n = original.len();
    debug_assert_eq!(transformed.len(), n);
    let minimum = 2 * usize::from(frequency.periods_per_year()) + 2;
    debug_assert!((minimum..=MAXIMUM_OBSERVATIONS).contains(&n));
    let mut tram = Vec::with_capacity(n);
    let mut logged = Vec::with_capacity(n);
    for (index, (&value, &transformed)) in original.iter().zip(transformed).enumerate() {
        debug_assert!(value.is_finite() && value > 0.0);
        // `transformed` is TRNFCN's output. ADJREG subtracts its ten
        // regression effects here, all exactly zero for this model; `x − 0.0`
        // is exactly `x` for every f64, −0.0 included, so they are left out.
        let orixa = transformed;
        let orixmv = transformed - 0.0; // The missing-value-adjusted series.
        let oz_value = orixa.exp(); // INVFCN(Stcsi) -> OZ.
        let tram_value = orixmv.exp(); // INVFCN(Orixs) -> Tram.
        for (quantity, rounded) in [
            ("SEATS input exp", oz_value),
            ("linearized series exp", tram_value),
        ] {
            usable_positive(rounded, quantity, Some(index))?;
        }
        let logged_value = oz_value.ln(); // ANALTS logs OZ again.
        finite(logged_value, "SEATS input log", Some(index))?;
        tram.push(tram_value);
        logged.push(logged_value);
    }
    Ok(LogInput {
        frequency,
        tram,
        logged,
    })
}

/// The final components of a log run, for a family without a transitory
/// component.
pub(crate) fn final_log(
    input: &LogInput,
    raw: &ExtractedComponents,
) -> Result<FinalComponents, Error> {
    debug_assert!(raw.cycle.is_none());
    Ok(final_log_parts(input, raw)?.0)
}

fn final_log_parts(input: &LogInput, raw: &ExtractedComponents) -> Result<LogComponents, Error> {
    final_log_with(
        input,
        raw,
        None,
        EffectiveTradingDay::Ordinary,
        |index, seasonal_percent, _| {
            // ADJREG also applies six multiplicative effects here, all exactly
            // one for this model; `x · 1.0` returns `x` for every f64, so they
            // are left out.
            let denominator = seasonal_percent / 100.0;
            finite(
                input.tram[index] / usable_positive(denominator, "saved SA divisor", Some(index))?,
                "saved SA",
                Some(index),
            )
        },
    )
}

/// The final components and SECOND's working cycle, in percent.
type LogComponents = (FinalComponents, Option<Vec<f64>>);

/// SECOND's conversion of the log components to the original scale. `adjust`
/// makes the seasonally adjusted value at an index from the seasonal and
/// cycle percents, and `effects` are the outlier effects multiplied into the
/// final trend and irregular.
pub(super) fn final_log_with<E: From<Error>>(
    input: &LogInput,
    raw: &ExtractedComponents,
    effects: Option<super::effects::FinalOutlierEffects<'_>>,
    effective: EffectiveTradingDay,
    mut adjust: impl FnMut(usize, f64, f64) -> Result<f64, E>,
) -> Result<LogComponents, E> {
    let n = input.logged.len();
    debug_assert_eq!(raw.frequency, input.frequency);
    debug_assert_eq!(input.tram.len(), n);
    debug_assert_eq!(raw.observed_len, n);
    debug_assert_eq!(raw.trend.len(), n);
    debug_assert_eq!(raw.seasonal.len(), n);
    debug_assert_eq!(raw.irregular.len(), n);
    debug_assert!(raw.cycle.as_ref().is_none_or(|cycle| cycle.len() == n));
    let periods = usize::from(input.frequency.periods_per_year());
    // The whole years of the span; a series holds at least two.
    let nyr = (n / periods) * periods;
    debug_assert!(nyr > 0);

    // SECOND's Bias1 loop sums the seasonal over the whole years and the
    // irregular over every observation.
    let mut bias1c = 0.0;
    let mut bias2c = 0.0;
    for (index, (&seasonal, &irregular)) in raw.seasonal.iter().zip(&raw.irregular).enumerate() {
        if index < nyr {
            bias1c += usable_positive(seasonal.exp(), "seasonal exp", Some(index))?;
            usable_positive(bias1c, "seasonal accumulation", Some(index))?;
        }
        bias2c += usable_positive(irregular.exp(), "irregular exp", Some(index))?;
        usable_positive(bias2c, "irregular accumulation", Some(index))?;
    }
    bias1c = usable_positive(bias1c / nyr as f64, "seasonal mean", None)?;
    bias2c = usable_positive(bias2c / n as f64, "irregular mean", None)?;
    let bias3c = usable_positive(bias1c * bias2c, "combined bias", None)?;

    let mut seasonal = Vec::with_capacity(n);
    let mut seasonally_adjusted = Vec::with_capacity(n);
    let mut trend = Vec::with_capacity(n);
    let mut irregular = Vec::with_capacity(n);
    let mut working_cycle = raw.cycle.as_ref().map(|_| Vec::with_capacity(n));
    // sigex passes IsCloseToTD to SECOND. For stochastic trading day,
    // sigsub.f:1569–1584 divides the series by seasonal*cycle and the result
    // by the trend; otherwise by the seasonal and then by trend*cycle.
    let td = effective == EffectiveTradingDay::Stochastic;
    for index in 0..n {
        let seasonal_factor = usable_positive(
            usable_positive(raw.seasonal[index].exp(), "seasonal exp", Some(index))? / bias1c,
            "seasonal factor",
            Some(index),
        )?;
        let cycle_factor = usable_positive(
            raw.cycle.as_ref().map_or(0.0, |c| c[index]).exp(),
            "cycle exp",
            Some(index),
        )?;
        let second_divisor = if td {
            usable_positive(
                seasonal_factor * cycle_factor,
                "trading-day SA divisor",
                Some(index),
            )?
        } else {
            seasonal_factor
        };
        let second_sa = usable_positive(
            usable_positive(input.logged[index].exp(), "logged exp", Some(index))? / second_divisor,
            "seasonally adjusted",
            Some(index),
        )?;
        let second_trend = usable_positive(
            usable_positive(raw.trend[index].exp(), "trend exp", Some(index))? * bias3c,
            "bias-corrected trend",
            Some(index),
        )?;
        let irregular_divisor = if td {
            second_trend
        } else {
            usable_positive(
                second_trend * cycle_factor,
                "irregular divisor",
                Some(index),
            )?
        };
        let irregular_ratio = usable_positive(
            second_sa / irregular_divisor,
            "irregular log argument",
            Some(index),
        )?;
        let irregular_log = finite(irregular_ratio.ln(), "irregular log", Some(index))?;
        let seasonal_percent = finite(100.0 * seasonal_factor, "seasonal percent", Some(index))?;
        let cycle_percent = finite(100.0 * cycle_factor, "cycle percent", Some(index))?;
        if let Some(values) = &mut working_cycle {
            values.push(cycle_percent);
        }
        let irregular_percent = finite(
            100.0 * irregular_log.exp(),
            "irregular percent",
            Some(index),
        )?;

        seasonal.push(finite(
            seasonal_percent / 100.0,
            "saved seasonal",
            Some(index),
        )?);
        seasonally_adjusted.push(adjust(index, seasonal_percent, cycle_percent)?);
        let mut final_trend = second_trend;
        for value in [effects.map_or(1.0, |e| e.trend[index]), 1.0, 1.0, 1.0] {
            final_trend *= value;
        }
        trend.push(finite(final_trend, SAVED_TREND, Some(index))?);
        let mut final_irregular = irregular_percent;
        for value in [effects.map_or(1.0, |e| e.irregular[index]), 1.0] {
            final_irregular *= value;
        }
        irregular.push(finite(
            final_irregular / 100.0,
            SAVED_IRREGULAR,
            Some(index),
        )?);
    }
    Ok((
        FinalComponents {
            seasonal,
            seasonally_adjusted,
            trend,
            irregular,
        },
        working_cycle,
    ))
}

/// `value`, or [`Failure::Unusable`] naming `quantity` when it is not finite.
pub(super) fn finite(
    value: f64,
    quantity: &'static str,
    index: Option<usize>,
) -> Result<f64, Error> {
    usable(quantity, index, value).map_err(|failure| failure.at(Stage::BiasCorrection))
}

/// The final components of a log run, for a family that can extract a
/// transitory component of variance `cycle_variance`.
pub(super) fn final_log_with_transitory(
    input: &LogInput,
    raw: &ExtractedComponents,
    cycle_variance: f64,
) -> Result<super::extraction::FinalWithTransitory, Error> {
    finite(cycle_variance, "saved cycle variance", None)?;
    let (saved, working_cycle) = final_log_parts(input, raw)?;
    let transitory = final_log_transitory(working_cycle, cycle_variance);
    Ok(super::extraction::FinalWithTransitory { saved, transitory })
}

/// The final transitory factors from SECOND's working cycle, or `None` when
/// DETCOMP leaves the component out.
pub(super) fn final_log_transitory(
    working_cycle: Option<Vec<f64>>,
    cycle_variance: f64,
) -> Option<Vec<f64>> {
    // DETCOMP keeps the cycle only when its variance exceeds 1e-10, whatever
    // SECOND's working cycle holds. It multiplies by Pareg5, which is one
    // here, before USRENTRY's `dpeq` scan, and seatad then divides by 100.
    if cycle_variance > 1e-10 {
        working_cycle
            .map(|values| values.into_iter().map(|v| v * 1.0).collect::<Vec<_>>())
            .filter(|values| super::extraction::component_present(values))
            .map(|values| values.into_iter().map(|v| v / 100.0).collect())
    } else {
        None
    }
}

/// `value`, or [`Failure::Unusable`] naming `quantity` when it is not finite
/// and positive.
fn usable_positive(value: f64, quantity: &'static str, index: Option<usize>) -> Result<f64, Error> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(Failure::Unusable {
            quantity,
            index,
            value,
        }
        .at(Stage::BiasCorrection))
    }
}

#[cfg(test)]
mod tests;
