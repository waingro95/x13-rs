//! MAspectrum's seasonally adjusted model (`spectrum.f`): the moving-average
//! factors of the canonical components and of the seasonally adjusted series,
//! and ShowComp's test of their innovation variances.

use super::factorization::{Factorization, MaFactor};
use super::harmonic::conjugate;
use super::polynomial::{SpectralInputs, convolve};
use crate::seats::failure::{
    CanonicalComponent, Component, Error, Failure, Stage, VarianceDirection,
};
use crate::seats::trading_day::EffectiveTradingDay;

/// MAspectrum's numerator of the seasonally adjusted spectrum: `base` times
/// the irregular variance `qt`, plus each component's harmonics times its
/// innovation variance.
///
/// The irregular product rounds on its own. The sum is padded with positive
/// zeros, and gfortran fuses the trend products into it, then the cycle
/// products in a second pass.
fn assemble_numerator(
    base: &[f64],
    qt: f64,
    trend: &[f64],
    variance: f64,
    cycle: Option<(&[f64], f64)>,
) -> Vec<f64> {
    let mut summed = base.iter().map(|v| v * qt).collect::<Vec<_>>();
    summed.resize(summed.len().max(trend.len()), 0.0);
    for (value, &coefficient) in summed.iter_mut().zip(trend) {
        *value = variance.mul_add(coefficient, *value);
    }
    if let Some((harmonic, var)) = cycle {
        summed.resize(summed.len().max(harmonic.len()), 0.0);
        for (value, &coefficient) in summed.iter_mut().zip(harmonic) {
            *value = var.mul_add(coefficient, *value);
        }
    }
    summed
}

/// The canonical components' factors with the seasonally adjusted series'.
#[derive(Debug)]
pub(crate) struct Ar1AdjustedModel {
    pub(crate) factors: Factorization,
    pub(crate) cycle: Option<MaFactor>,
    pub(crate) inputs: SpectralInputs,
    pub(crate) irregular_candidate: f64,
    pub(super) adjusted: MaFactor,
}

/// MAspectrum's factors for a model whose irregular variance is not negative.
/// A negative one is an error, which the run answers by changing the model.
pub(crate) fn factor_adjusted(
    preparation: super::canonical::CanonicalPreparation,
    inputs: SpectralInputs,
    trading_day: EffectiveTradingDay,
) -> Result<Ar1AdjustedModel, Error> {
    let qt = preparation.minima.irregular_candidate;
    valid_value(qt, "irregular variance", qt.is_finite())?;
    if qt < 0.0 {
        return Err(Failure::NegativeVariance {
            component: CanonicalComponent::Irregular,
            variance: qt,
        }
        .at(Stage::CanonicalDecomposition));
    }
    factor_components(preparation, inputs, trading_day)
}

/// MAspectrum's factors for a model whose irregular variance is negative,
/// which ShowInvalDecomp reports.
pub(crate) fn factor_rejected_ar2_with_trading_day(
    preparation: super::canonical::CanonicalPreparation,
    inputs: SpectralInputs,
    trading_day: EffectiveTradingDay,
) -> Result<Ar1AdjustedModel, Error> {
    let qt = preparation.minima.irregular_candidate;
    valid_value(qt, "irregular variance", qt.is_finite() && qt < 0.0)?;
    factor_components(preparation, inputs, trading_day)
}

fn factor_components(
    preparation: super::canonical::CanonicalPreparation,
    inputs: SpectralInputs,
    trading_day: EffectiveTradingDay,
) -> Result<Ar1AdjustedModel, Error> {
    let qt = preparation.minima.irregular_candidate;
    let factor = super::factorization::factor_numerator;
    let trend = if inputs.trend.combined.len() == 1 {
        // DecompSpectrum skips MAK1 for nchi=1 and keeps THETP=[1] with a
        // zero trend innovation variance.
        MaFactor::absent()
    } else {
        factor(&preparation.trend.numerator)?
    };
    let seasonal = if inputs.seasonal.combined.len() == 1 {
        // DecompSpectrum starts from THETS=[1] and skips MAK1 for npsi=1.
        MaFactor::absent()
    } else {
        factor(&preparation.seasonal.numerator)?
    };
    let cycle = preparation
        .cycle
        .as_ref()
        .map(|c| factor(&c.numerator))
        .transpose()?;
    let factors = Factorization {
        preparation,
        trend,
        seasonal,
    };
    let harmonics = &factors.preparation.minima.harmonics;
    let ncycth = harmonics.numerator.len() > harmonics.denominator.len();
    let adjusted = if inputs.trend_cycle.len() == 1 && !ncycth {
        // With neither trend nor cycle (nchcyc=1 and ncycth=0) MAspectrum
        // skips the whole block, the Npsi=1 copy included. THADJ keeps
        // DecompSpectrum's [1] and VARWNA stays zero.
        MaFactor::absent()
    } else if inputs.seasonal.combined.len() == 1 {
        // With Npsi=1 MAspectrum copies the aggregate MA, with unit
        // variance, instead of factoring the assembled spectrum.
        MaFactor {
            coefficients: inputs.aggregate_ma.clone(),
            variance: 1.0,
        }
    } else {
        let base = conjugate(&inputs.trend_cycle, &inputs.trend_cycle);
        // With a stochastic trading day MAspectrum uses Thetp alone and
        // leaves the cycle out of the seasonally adjusted series.
        let trend_harmonic = match trading_day {
            EffectiveTradingDay::Stochastic => {
                conjugate(&factors.trend.coefficients, &factors.trend.coefficients)
            }
            EffectiveTradingDay::Ordinary => {
                let product = convolve(&factors.trend.coefficients, &inputs.cycle.combined);
                conjugate(&product, &product)
            }
        };
        let cycle_harmonic = cycle
            .as_ref()
            .filter(|_| trading_day == EffectiveTradingDay::Ordinary)
            .map(|factor| {
                let product = convolve(&factor.coefficients, &inputs.trend.combined);
                conjugate(&product, &product)
            });
        let numerator = assemble_numerator(
            &base,
            qt,
            &trend_harmonic,
            factors.trend.variance,
            cycle_harmonic
                .as_deref()
                .zip(cycle.as_ref().map(|c| c.variance)),
        );
        if let Some(index) = numerator.iter().position(|value| !value.is_finite()) {
            return Err(Failure::Unusable {
                quantity: "spectral numerator assembly",
                index: Some(index),
                value: numerator[index],
            }
            .at(Stage::CanonicalDecomposition));
        }
        if numerator[0] <= 0.0 {
            return Err(Failure::UnusableNumerator {
                zero_lag: numerator[0],
            }
            .at(Stage::CanonicalDecomposition));
        }
        let adjusted = factor(&numerator)?;
        valid_value(
            adjusted.variance,
            "adjusted variance",
            adjusted.variance.is_finite() && adjusted.variance > 0.0,
        )?;
        adjusted
    };
    Ok(Ar1AdjustedModel {
        factors,
        cycle,
        inputs,
        irregular_candidate: qt,
        adjusted,
    })
}

/// ShowComp's component order: trend, seasonal, transitory (only when
/// present), then SA. An absent trend or seasonal keeps DecompSpectrum's zero
/// variance, so it always passes.
pub(crate) fn show_comp_variances(model: &Ar1AdjustedModel) -> [Option<(Component, f64)>; 4] {
    [
        Some((Component::Trend, model.factors.trend.variance)),
        Some((Component::Seasonal, model.factors.seasonal.variance)),
        model
            .cycle
            .as_ref()
            .map(|cycle| (Component::Transitory, cycle.variance)),
        Some((Component::SeasonallyAdjusted, model.adjusted.variance)),
    ]
}

/// ShowComp's refusal test, `VARWN.GT.ONE.OR.VARWN.LT.ZERO`.
pub(crate) fn show_comp_direction(variance: f64) -> Option<VarianceDirection> {
    if variance < 0.0 {
        Some(VarianceDirection::BelowZero)
    } else if variance > 1.0 {
        Some(VarianceDirection::AboveOne)
    } else {
        None
    }
}

/// ShowComp closes DecompSpectrum, after its negative-transitory test: a
/// component's innovation variance (in units of var(a)) above one or below
/// zero sets Lfatal and writes `seatsadj: no`, so SEATS stops. The families
/// built on `Ar1AdjustedModel` (ARMA, AR1 with seasonal MA, second
/// difference) call this before extraction. The AR2 family applies the same
/// test, sharing the component list and comparison.
///
/// Build 62 gates the trend test on OUT=0 and runs ShowComp only in the main
/// pass. Neither option exists here, so all four tests always run.
pub(crate) fn check_show_comp(model: &Ar1AdjustedModel) -> Result<(), Error> {
    for (component, variance) in show_comp_variances(model).into_iter().flatten() {
        if let Some(direction) = show_comp_direction(variance) {
            return Err(Failure::InnovationVariance {
                component,
                direction,
                value: variance,
            }
            .at(Stage::CanonicalDecomposition));
        }
    }
    Ok(())
}

fn valid_value(value: f64, quantity: &'static str, valid: bool) -> Result<(), Error> {
    if !valid {
        return Err(Failure::Unusable {
            quantity,
            index: None,
            value,
        }
        .at(Stage::CanonicalDecomposition));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
