//! ESTBUR's signal extraction and the final components DETCOMP makes of it.
//!
//! ESTBUR (`ansub3.f`) estimates the trend, the seasonal, the transitory
//! component (its "cycle") and the irregular with the Burman-Wilson form of
//! the Wiener-Kolmogorov filter. It solves each component filter's partition
//! over the model's MA, filters the series FCAST extended in both directions,
//! solves the endpoint equations and runs the backward recurrences. [`extract`]
//! is that body for every family. The airline and the seasonal single MA
//! solve systems of a size the frequency fixes; the other families size
//! theirs from the aggregate MA.
//!
//! `final_identity` and its variants follow DETCOMP and USRENTRY for a series
//! without a transformation: the seasonally adjusted series is the original
//! less the seasonal, and a component that is zero throughout is left out.

use std::borrow::Cow;

use super::airline::model::ResolvedAirline;
use super::decomposition::canonical::{CanonicalPreparation, prepare_canonical};
use super::decomposition::minima::spectral_minima;
use super::decomposition::polynomial::{SpectralInputs, spectral_inputs};
use super::failure::{CanonicalComponent, Error, Failure, Stage};
use super::linear::{EstburSystem, solve, solve_estbur};
use super::tails::{
    DifferenceOrder, SeriesTails, airline_bj_ma, ar1_bj_ma, bphist_ordered, resolve_horizons,
    single_lag_bj_ma,
};
use crate::Frequency;
use crate::numerics::RowMajorMatrix;

/// The label of an unusable saved trend.
pub(super) const SAVED_TREND: &str = "saved trend";
/// The label of an unusable saved irregular.
pub(super) const SAVED_IRREGULAR: &str = "saved irregular";

/// The ESTBUR or DETCOMP value a failure names.
#[derive(Clone, Copy)]
enum Quantity {
    SavedSa,
    SavedTrend,
    SavedIrregular,
    SavedCycleVariance,
    SavedCycle,
    ForwardBjMa,
    BackwardBjMa,
    AggregateDenominator,
    AggregateMa,
    Observed,
    ImportedAr,
    TrendFilter,
    SeasonalFilter,
    CycleFilter,
    StationaryDmeanSum,
    StationaryDmean,
    StationaryCenteredExtension,
    ObservedTrend,
    StationaryRestoredTrend,
    ObservedSeasonal,
    ObservedCycle,
    ObservedIrregular,
    PartitionMatrix,
    ForwardTrendSum,
    BackwardTrendSum,
    ForwardSeasonalSum,
    BackwardSeasonalSum,
    ForwardCycleSum,
    BackwardCycleSum,
    BackwardRecurrence,
}

impl Quantity {
    const fn label(self) -> &'static str {
        match self {
            Self::SavedSa => "saved SA",
            Self::SavedTrend => SAVED_TREND,
            Self::SavedIrregular => SAVED_IRREGULAR,
            Self::SavedCycleVariance => "saved cycle variance",
            Self::SavedCycle => "saved cycle",
            Self::ForwardBjMa => "forward Box-Jenkins MA",
            Self::BackwardBjMa => "backward Box-Jenkins MA",
            Self::AggregateDenominator => "aggregate denominator",
            Self::AggregateMa => "aggregate MA",
            Self::Observed => "observed value",
            Self::ImportedAr => "imported AR coefficient",
            Self::TrendFilter => "canonical trend filter",
            Self::SeasonalFilter => "canonical seasonal filter",
            Self::CycleFilter => "canonical cycle filter",
            Self::StationaryDmeanSum => "stationary mean sum",
            Self::StationaryDmean => "stationary mean",
            Self::StationaryCenteredExtension => "stationary centered extension",
            Self::ObservedTrend => "observed trend",
            Self::StationaryRestoredTrend => "stationary restored trend",
            Self::ObservedSeasonal => "observed seasonal",
            Self::ObservedCycle => "observed cycle",
            Self::ObservedIrregular => "observed irregular",
            Self::PartitionMatrix => "partition matrix",
            Self::ForwardTrendSum => "forward trend sum",
            Self::BackwardTrendSum => "backward trend sum",
            Self::ForwardSeasonalSum => "forward seasonal sum",
            Self::BackwardSeasonalSum => "backward seasonal sum",
            Self::ForwardCycleSum => "forward cycle sum",
            Self::BackwardCycleSum => "backward cycle sum",
            Self::BackwardRecurrence => "backward recurrence",
        }
    }
}

/// ESTBUR's estimates over the observed span. `cycle` is the transitory
/// component, which only the families with an AR part extract.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExtractedComponents {
    pub(crate) frequency: Frequency,
    pub(crate) observed_len: usize,
    pub(crate) trend: Vec<f64>,
    pub(crate) seasonal: Vec<f64>,
    pub(crate) irregular: Vec<f64>,
    pub(crate) cycle: Option<Vec<f64>>,
}

/// The final components. `trend` is empty when DETCOMP's trend is zero
/// throughout.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalComponents {
    pub(crate) seasonal: Vec<f64>,
    pub(crate) seasonally_adjusted: Vec<f64>,
    pub(crate) trend: Vec<f64>,
    pub(crate) irregular: Vec<f64>,
}

/// The final components of an untransformed series, for a family without a
/// transitory component.
pub(crate) fn final_identity(
    original: &[f64],
    raw: &ExtractedComponents,
) -> Result<FinalComponents, Error> {
    debug_assert!(raw.cycle.is_none());
    final_identity_parts(original, raw)
}

fn final_identity_parts(
    original: &[f64],
    raw: &ExtractedComponents,
) -> Result<FinalComponents, Error> {
    final_identity_with::<Error>(original, raw, None, |i, sc| {
        // DETCOMP adds five zero effects to the seasonal. Only the first can
        // change its bits (it turns -0 into +0), so one addition stands in.
        finite(original[i] - (sc + 0.0), Quantity::SavedSa, Some(i))
    })
}

/// DETCOMP's final components. `adjust` makes the seasonally adjusted value
/// at an index from its seasonal, and `effects` are the outlier effects
/// DETCOMP adds back to the trend and the irregular.
pub(super) fn final_identity_with<E: From<Error>>(
    original: &[f64],
    raw: &ExtractedComponents,
    effects: Option<super::effects::FinalOutlierEffects<'_>>,
    mut adjust: impl FnMut(usize, f64) -> Result<f64, E>,
) -> Result<FinalComponents, E> {
    debug_assert_eq!(original.len(), raw.observed_len);
    debug_assert_eq!(raw.seasonal.len(), raw.observed_len);
    debug_assert_eq!(raw.trend.len(), raw.observed_len);
    debug_assert_eq!(raw.irregular.len(), raw.observed_len);
    let mut seasonal = Vec::with_capacity(raw.observed_len);
    let mut seasonally_adjusted = Vec::with_capacity(raw.observed_len);
    let mut trend = Vec::with_capacity(raw.observed_len);
    let mut irregular = Vec::with_capacity(raw.observed_len);
    for i in 0..original.len() {
        // DETCOMP (ansub4.f) leaves S10 as the seasonal USRENTRY 1201 saved.
        seasonal.push(raw.seasonal[i]);
        seasonally_adjusted.push(adjust(i, raw.seasonal[i])?);
        let mut t_effect = raw.trend[i];
        for value in [effects.map_or(0.0, |e| e.trend[i]), 0.0, 0.0] {
            t_effect += value;
        }
        trend.push(finite(t_effect, Quantity::SavedTrend, Some(i))?);
        let mut ir_effect = raw.irregular[i];
        for value in [effects.map_or(0.0, |e| e.irregular[i]), 0.0] {
            ir_effect += value;
        }
        irregular.push(finite(ir_effect, Quantity::SavedIrregular, Some(i))?);
    }
    // USRENTRY 1310 (ansub9.f) tests the final DETCOMP trend, restored trend
    // effects included, and seatpr omits S12 when `dpeq` finds it all zero.
    if !component_present(&trend) {
        trend.clear();
    }
    Ok(FinalComponents {
        seasonal,
        seasonally_adjusted,
        trend,
        irregular,
    })
}

/// The final components with the transitory component, where the family has
/// one that is not zero throughout.
#[derive(Debug)]
pub(super) struct FinalWithTransitory {
    pub(super) saved: FinalComponents,
    pub(super) transitory: Option<Vec<f64>>,
}

/// The final components of an untransformed series, for a family that can
/// extract a transitory component of variance `cycle_variance`.
pub(super) fn final_identity_with_transitory(
    original: &[f64],
    raw: &ExtractedComponents,
    cycle_variance: f64,
) -> Result<FinalWithTransitory, Error> {
    let saved = final_identity_parts(original, raw)?;
    let transitory = final_identity_transitory(raw, cycle_variance)?;
    Ok(FinalWithTransitory { saved, transitory })
}

/// DETCOMP's saved transitory component, after the check of its variance.
pub(super) fn final_identity_transitory(
    raw: &ExtractedComponents,
    cycle_variance: f64,
) -> Result<Option<Vec<f64>>, Error> {
    finite(cycle_variance, Quantity::SavedCycleVariance, None)?;
    let transitory = raw
        .cycle
        .as_ref()
        .map(|cycle| {
            debug_assert_eq!(cycle.len(), raw.observed_len);
            cycle
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    // DETCOMP replaces the cycle by cycle+Pareg5 only when its
                    // variance exceeds 1e-10. Pareg5 is zero here, so the sum
                    // can only turn -0 into +0.
                    finite(
                        if cycle_variance > 1e-10 {
                            value + 0.0
                        } else {
                            value
                        },
                        Quantity::SavedCycle,
                        Some(i),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .filter(|values| component_present(values));
    Ok(transitory)
}

/// Whether a final component has a value that Build 62's `dpeq` tells from
/// zero. USRENTRY 1310 and 1313 (ansub9.f) save a component only then.
pub(super) fn component_present(values: &[f64]) -> bool {
    values.iter().any(|value| value.abs() >= 3.834e-20)
}

fn same_bits(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| a.to_bits() == e.to_bits())
}

pub(super) fn extract_ar1(
    model: &super::ar1_seasonal_ma::model::ResolvedAr1SeasonalMa,
    inputs: &super::decomposition::polynomial::SpectralInputs,
    canonical: &super::decomposition::canonical::CanonicalPreparation,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    extract(
        ExtractionModel::Ar1 {
            model,
            inputs,
            canonical,
        },
        observed,
        tails,
    )
}

pub(super) fn extract_arma<const N: usize>(
    frequency: Frequency,
    inputs: &super::decomposition::polynomial::SpectralInputs,
    canonical: &super::decomposition::canonical::CanonicalPreparation,
    observed: &[f64],
    tails: &super::arma::ArmaRunTails<N>,
    mean: super::mean::RequestedMean,
) -> Result<ExtractedComponents, Error> {
    let s = usize::from(frequency.periods_per_year());
    let qstar = if tails.backward_coefficients.layout.has_seasonal_ma() {
        s + tails.backward_coefficients.layout.regular_ma_count()
    } else {
        tails.backward_coefficients.layout.regular_ma_count()
    };
    for (stage, values) in [
        (Quantity::ForwardBjMa, tails.base.bj_ma.as_slice()),
        (Quantity::BackwardBjMa, tails.backward_bj_ma.as_slice()),
    ] {
        debug_assert_eq!(values.len(), qstar);
        for (index, &value) in values.iter().enumerate() {
            finite(value, stage, Some(index))?;
        }
    }
    // Each Box-Jenkins MA is the THSTAR of its CALCFX, and the backward one
    // is the aggregate MA with its signs reversed.
    debug_assert!(same_bits(
        &tails.base.bj_ma,
        tails.forward_coefficients.thstar.as_slice()
    ));
    debug_assert!(same_bits(
        &tails.backward_bj_ma,
        tails.backward_coefficients.thstar.as_slice()
    ));
    debug_assert_eq!(inputs.aggregate_ma.len(), qstar + 1);
    debug_assert!(
        tails
            .backward_bj_ma
            .iter()
            .zip(&inputs.aggregate_ma[1..])
            .all(|(&actual, &coefficient)| actual == -coefficient)
    );
    extract(
        ExtractionModel::Arma {
            frequency,
            inputs,
            canonical,
            center_mean: tails.backward_coefficients.layout.differences()
                == DifferenceOrder::Stationary
                && mean == super::mean::RequestedMean::Yes,
        },
        observed,
        &tails.base,
    )
}

pub(super) fn extract_airline(
    model: &ResolvedAirline,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    extract(ExtractionModel::Airline(model), observed, tails)
}

/// The airline's canonical decomposition from SPECTRUM's minima. With
/// NOADMISS=0, the default, SPECTRUM rejects a model whose irregular spectrum
/// is negative.
fn airline_canonical(
    minima: super::decomposition::minima::SpectralMinima,
) -> Result<CanonicalPreparation, Error> {
    if minima.irregular_candidate < 0.0 {
        return Err(negative_irregular(minima.irregular_candidate));
    }
    Ok(prepare_canonical(minima))
}

/// SPECTRUM's refusal of a negative irregular spectrum.
fn negative_irregular(variance: f64) -> Error {
    Failure::NegativeVariance {
        component: CanonicalComponent::Irregular,
        variance,
    }
    .at(Stage::Extraction)
}

/// What ESTBUR reads of a family's model.
#[derive(Clone, Copy)]
enum ExtractionModel<'a> {
    Ar2(&'a super::ar2::canonical::Ar2Preparation),
    Airline(&'a ResolvedAirline),
    SeasonalMa(&'a super::seasonal_ma::ResolvedSeasonalMa),
    SecondDifference {
        model: &'a super::second_difference::model::ResolvedSecondDifference,
        inputs: &'a super::decomposition::polynomial::SpectralInputs,
        canonical: &'a super::decomposition::canonical::CanonicalPreparation,
    },
    Ar1 {
        model: &'a super::ar1_seasonal_ma::model::ResolvedAr1SeasonalMa,
        inputs: &'a super::decomposition::polynomial::SpectralInputs,
        canonical: &'a super::decomposition::canonical::CanonicalPreparation,
    },
    Arma {
        frequency: Frequency,
        inputs: &'a super::decomposition::polynomial::SpectralInputs,
        canonical: &'a super::decomposition::canonical::CanonicalPreparation,
        center_mean: bool,
    },
}

impl ExtractionModel<'_> {
    fn frequency(&self) -> Frequency {
        match self {
            Self::Ar2(preparation) => preparation.frequency,
            Self::Airline(model) => model.frequency(),
            Self::SeasonalMa(model) => model.frequency,
            Self::SecondDifference { model, .. } => model.frequency,
            Self::Ar1 { model, .. } => model.frequency(),
            Self::Arma { frequency, .. } => *frequency,
        }
    }
    /// The MA family whose ESTBUR systems have sizes the frequency fixes: the
    /// airline and the seasonal single MA. Every other family takes its MA
    /// length from the aggregate polynomial and sizes its systems from it.
    fn fixed_size_family(&self) -> Option<super::decomposition::polynomial::MaFamily> {
        match self {
            Self::Airline(_) => Some(super::decomposition::polynomial::MaFamily::Airline),
            Self::SeasonalMa(_) if !self.is_nonseasonal_ma() => {
                Some(super::decomposition::polynomial::MaFamily::SeasonalMa)
            }
            Self::Ar2(_)
            | Self::SeasonalMa(_)
            | Self::SecondDifference { .. }
            | Self::Ar1 { .. }
            | Self::Arma { .. } => None,
        }
    }
    /// True for AR2, AR1 with seasonal MA, and the generic ARMA family,
    /// whose layouts include pure-MA ones such as (0,1,2)(0,1,1). Only they
    /// extract a cycle. The match is exhaustive so that a new family must be
    /// classified.
    fn is_general_arma(&self) -> bool {
        match self {
            Self::Ar2(_) | Self::Ar1 { .. } | Self::Arma { .. } => true,
            Self::Airline(_) | Self::SeasonalMa(_) | Self::SecondDifference { .. } => false,
        }
    }
    fn is_nonseasonal_ma(&self) -> bool {
        matches!(self, Self::SeasonalMa(model) if model.shape == super::seasonal_ma::SingleMaShape::Nonseasonal)
    }
    /// FCAST's working forecast length: FHI, or more where the family's
    /// recurrence needs more values.
    fn working_horizon(&self, horizon: usize) -> usize {
        let s = usize::from(self.frequency().periods_per_year());
        match self {
            Self::Ar2(preparation) => horizon.max(2 * s + preparation.ar.as_slice().len() + 1),
            Self::SecondDifference { .. } => horizon.max(2 * s + 3),
            Self::Arma { inputs, .. } => {
                // FCAST:2154–2161 writes L=max(bpstar+Qstar,FHI) forecasts,
                // while SIGEX consumes only FHI. Cubic ARMA quarterly models
                // need one additional working value beyond the 3-year horizon.
                horizon.max(inputs.aggregate_denominator.len() + inputs.aggregate_ma.len() - 2)
            }
            Self::Airline(_) | Self::SeasonalMa(_) | Self::Ar1 { .. } => horizon,
        }
    }
    /// The Box-Jenkins MA this family's tails carry. ARMA has none to
    /// rebuild here: `extract_arma` compares its tails with CALCFX's THSTAR.
    fn bj_ma(&self) -> Option<Vec<f64>> {
        let s = usize::from(self.frequency().periods_per_year());
        match self {
            Self::Ar2(preparation) => Some(single_lag_bj_ma(s, preparation.seasonal_ma)),
            Self::Airline(model) => Some(airline_bj_ma(s, model.ma_coefficients())),
            Self::SeasonalMa(model) => Some(single_lag_bj_ma(
                model.shape.lag(model.frequency),
                model.seasonal_ma,
            )),
            Self::SecondDifference { model, .. } => Some(airline_bj_ma(s, model.parameters)),
            Self::Ar1 { model, .. } => Some(ar1_bj_ma(model)),
            Self::Arma { .. } => None,
        }
    }
}

pub(super) fn extract_ar2(
    preparation: &super::ar2::canonical::Ar2Preparation,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    let s = usize::from(preparation.frequency.periods_per_year());
    debug_assert_eq!(
        preparation.model.inputs.aggregate_denominator.len(),
        s + preparation.ar.as_slice().len() + 2
    );
    debug_assert_eq!(preparation.model.inputs.aggregate_ma.len(), s + 1);
    for (stage, values) in [
        (Quantity::Observed, observed),
        (Quantity::ImportedAr, preparation.ar.as_slice()),
        (
            Quantity::AggregateDenominator,
            preparation.model.inputs.aggregate_denominator.as_slice(),
        ),
        (
            Quantity::AggregateMa,
            preparation.model.inputs.aggregate_ma.as_slice(),
        ),
        (
            Quantity::TrendFilter,
            preparation
                .model
                .factors
                .preparation
                .trend
                .filter_lags
                .as_slice(),
        ),
        (
            Quantity::SeasonalFilter,
            preparation
                .model
                .factors
                .preparation
                .seasonal
                .filter_lags
                .as_slice(),
        ),
    ] {
        for (index, &value) in values.iter().enumerate() {
            finite(value, stage, Some(index))?;
        }
    }
    if let Some(cycle) = &preparation.model.factors.preparation.cycle {
        for (index, &value) in cycle.filter_lags.iter().enumerate() {
            finite(value, Quantity::CycleFilter, Some(index))?;
        }
    }
    // The aggregate MA is the seasonal MA alone; a zero of either sign may
    // stand at the lags between.
    debug_assert!(
        preparation
            .model
            .inputs
            .aggregate_ma
            .iter()
            .enumerate()
            .all(|(index, &actual)| {
                actual
                    == if index == 0 {
                        1.0
                    } else if index == s {
                        -preparation.seasonal_ma
                    } else {
                        0.0
                    }
            })
    );
    debug_assert!(same_bits(
        &tails.bphist,
        &bphist_ordered(s, &preparation.ar.polynomial(), DifferenceOrder::First)
    ));
    extract(ExtractionModel::Ar2(preparation), observed, tails)
}

pub(super) fn extract_seasonal_ma(
    model: &super::seasonal_ma::ResolvedSeasonalMa,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    extract(ExtractionModel::SeasonalMa(model), observed, tails)
}

pub(super) fn extract_second_difference(
    model: &super::second_difference::model::ResolvedSecondDifference,
    inputs: &super::decomposition::polynomial::SpectralInputs,
    canonical: &super::decomposition::canonical::CanonicalPreparation,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    extract(
        ExtractionModel::SecondDifference {
            model,
            inputs,
            canonical,
        },
        observed,
        tails,
    )
}

/// ESTBUR for one family: the components of `observed` from its FCAST
/// `tails`.
fn extract(
    model: ExtractionModel<'_>,
    observed: &[f64],
    tails: &SeriesTails,
) -> Result<ExtractedComponents, Error> {
    debug_assert!(tails_match(model, observed, tails));
    let (inputs, canonical) = spectral_preparation(model)?;
    let s = usize::from(model.frequency().periods_per_year());
    let n = observed.len();
    let p = inputs.aggregate_denominator.len(); // ESTBUR pstar, including constant.
    let q = model
        .fixed_size_family()
        .map_or(inputs.aggregate_ma.len(), |family| {
            family.ma_coefficients(s)
        });
    let mut thstar = inputs.aggregate_ma.clone();
    thstar.resize(p.max(q), 0.0); // ESTBUR pads only when pstar exceeds qstar.
    let gains = partition_gains(model, &thstar, &canonical)?;

    // ESTBUR (ANSUB3:121–133): stationary models with a mean center
    // both extended directions and disable endpoint mean forcing.
    let center_mean = matches!(
        model,
        ExtractionModel::Arma {
            center_mean: true,
            ..
        }
    );
    let zmean = if center_mean {
        let mut sum = 0.0;
        for &value in observed {
            sum = finite(sum + value, Quantity::StationaryDmeanSum, None)?;
        }
        finite(sum / n as f64, Quantity::StationaryDmean, None)?
    } else {
        0.0
    };
    let centered = |values: &[f64]| -> Result<Vec<f64>, Error> {
        values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                finite(
                    value - zmean,
                    Quantity::StationaryCenteredExtension,
                    Some(i),
                )
            })
            .collect()
    };
    let (forward, backward, zaf, zab) = if center_mean {
        (
            Cow::Owned(centered(&tails.forward_extended)?),
            Cow::Owned(centered(&tails.backward_extended)?),
            0.0,
            0.0,
        )
    } else {
        (
            Cow::Borrowed(tails.forward_extended.as_slice()),
            Cow::Borrowed(tails.backward_extended.as_slice()),
            tails.zaf,
            tails.zab,
        )
    };

    let y = filtered_sums(&gains, &forward, &backward, n + q - 1)?;
    let x = endpoint_recurrence(
        model,
        &y,
        &thstar[..q],
        &inputs.aggregate_denominator,
        zaf,
        zab,
    )?;

    let mut trend = Vec::with_capacity(n);
    let mut seasonal = Vec::with_capacity(n);
    let mut irregular = Vec::with_capacity(n);
    let mut cycles =
        (model.is_general_arma() && canonical.cycle.is_some()).then(|| Vec::with_capacity(n));
    for (i, &value) in observed.iter().enumerate() {
        // Each component adds its forward run at i to its backward run at
        // the mirrored index.
        let (here, mirrored) = (x[i], x[n - i - 1]);
        let t = finite(
            here.trend.forward + mirrored.trend.backward,
            Quantity::ObservedTrend,
            Some(i),
        )?;
        // ESTBUR:490–500 restores the mean before component subtraction.
        let t = if center_mean {
            finite(t + zmean, Quantity::StationaryRestoredTrend, Some(i))?
        } else {
            t
        };
        let sc = finite(
            here.seasonal.forward + mirrored.seasonal.backward,
            Quantity::ObservedSeasonal,
            Some(i),
        )?;
        let cycle = finite(
            here.cycle.forward + mirrored.cycle.backward,
            Quantity::ObservedCycle,
            Some(i),
        )?;
        // ansub3.f:527–534 also forms SA here (less the working cycle under
        // AR2 stochastic TD). The final components recompute SA from the
        // seasonal, so that value is not kept.
        let ir = finite(
            ((value - sc) - t) - cycle,
            Quantity::ObservedIrregular,
            Some(i),
        )?;
        trend.push(t);
        seasonal.push(sc);
        irregular.push(ir);
        if let Some(values) = &mut cycles {
            values.push(cycle);
        }
    }
    Ok(ExtractedComponents {
        frequency: model.frequency(),
        observed_len: n,
        trend,
        seasonal,
        irregular,
        cycle: cycles,
    })
}

/// Whether `tails` are this model's FCAST tails of `observed`: sized to its
/// horizons, holding the series forward and reversed, with the family's
/// Box-Jenkins MA.
fn tails_match(model: ExtractionModel<'_>, observed: &[f64], tails: &SeriesTails) -> bool {
    let n = observed.len();
    let horizons = resolve_horizons(model.frequency(), n);
    let extended_len = n + model.working_horizon(horizons.fhi);
    tails.frequency == model.frequency()
        && tails.observed_len == n
        && tails.horizon == horizons.fhi
        && tails.forward_extended.len() == extended_len
        && tails.backward_extended.len() == extended_len
        && same_bits(&tails.forward_extended[..n], observed)
        && tails.backward_extended[..n]
            .iter()
            .zip(observed.iter().rev())
            .all(|(actual, value)| actual.to_bits() == value.to_bits())
        && model
            .bj_ma()
            .is_none_or(|expected| same_bits(&tails.bj_ma, &expected))
}

/// The aggregate polynomials and canonical filters ESTBUR partitions. The
/// prepared families carry them; the airline and the single MA run
/// SPECTRUM here.
fn spectral_preparation<'a>(
    model: ExtractionModel<'a>,
) -> Result<(Cow<'a, SpectralInputs>, Cow<'a, CanonicalPreparation>), Error> {
    match model {
        ExtractionModel::Ar2(preparation) => Ok((
            Cow::Borrowed(&preparation.model.inputs),
            Cow::Borrowed(&preparation.model.factors.preparation),
        )),
        ExtractionModel::Airline(model) => {
            let inputs = spectral_inputs(model);
            // SPECTRUM's PARFRA solve.
            let minima =
                spectral_minima(model).map_err(|error| error.restage(Stage::Extraction))?;
            let canonical = airline_canonical(minima)?;
            Ok((Cow::Owned(inputs), Cow::Owned(canonical)))
        }
        ExtractionModel::SeasonalMa(model) => {
            let inputs = super::decomposition::polynomial::seasonal_ma_spectral_inputs(model);
            let prefix = if model.shape == super::seasonal_ma::SingleMaShape::Nonseasonal {
                Ok(super::decomposition::rational::nonseasonal_ma_prefix(
                    &inputs,
                ))
            } else {
                super::decomposition::rational::rational_prefix_inputs(
                    &inputs,
                    super::decomposition::polynomial::MaFamily::SeasonalMa,
                )
            }
            .map_err(|error| error.restage(Stage::Extraction))?;
            let minima =
                super::decomposition::minima::spectral_minima_prefix(model.frequency, prefix);
            if minima.irregular_candidate < 0.0 {
                return Err(negative_irregular(minima.irregular_candidate));
            }
            Ok((Cow::Owned(inputs), Cow::Owned(prepare_canonical(minima))))
        }
        ExtractionModel::Ar1 {
            inputs, canonical, ..
        }
        | ExtractionModel::Arma {
            inputs, canonical, ..
        }
        | ExtractionModel::SecondDifference {
            inputs, canonical, ..
        } => Ok((Cow::Borrowed(inputs), Cow::Borrowed(canonical))),
    }
}

/// One lag of ESTBUR's trend, seasonal and cycle partition filters (GT, GS
/// and GC).
#[derive(Clone, Copy)]
struct Gains {
    trend: f64,
    seasonal: f64,
    cycle: f64,
}

/// ESTBUR's partition (ansub3.f:133–170): MAX(pstar,qstar) rows,
/// independently of endpoint size, solved for the filters at each lag.
fn partition_gains(
    model: ExtractionModel<'_>,
    thstar: &[f64],
    canonical: &CanonicalPreparation,
) -> Result<Vec<Gains>, Error> {
    let maxpq = thstar.len();
    let ct = &canonical.trend.filter_lags;
    let cs = &canonical.seasonal.filter_lags;
    let zero_cycle = [0.0; 32];
    let cc = canonical
        .cycle
        .as_ref()
        .map_or(&zero_cycle, |cycle| &cycle.filter_lags);
    // Row r pairs with lag k = maxpq - 1 - r.
    let columns = maxpq + 3;
    let mut partition = vec![0.0; maxpq * columns];
    for (r, row) in partition.chunks_exact_mut(columns).enumerate() {
        let k = maxpq - 1 - r;
        for (cell, &coefficient) in row[..=r].iter_mut().zip(thstar[..=r].iter().rev()) {
            *cell = coefficient;
        }
        for (cell, &coefficient) in row[k..maxpq].iter_mut().zip(thstar.iter().rev()) {
            *cell = finite(*cell + coefficient, Quantity::PartitionMatrix, Some(r))?;
        }
        row[maxpq..].copy_from_slice(&[ct[k], cs[k], cc[k]]);
    }
    let partition = RowMajorMatrix::new(partition, columns);
    let partition = match model.fixed_size_family() {
        Some(_) => solve_estbur(model.frequency(), EstburSystem::Partition, partition),
        None => solve(partition, maxpq, 3),
    }
    .map_err(|failure| failure.at(Stage::Extraction))?;
    // The last row holds lag 0.
    Ok(partition
        .rows()
        .rev()
        .map(|row| Gains {
            trend: row[maxpq],
            seasonal: row[maxpq + 1],
            cycle: row[maxpq + 2],
        })
        .collect())
}

/// A forward and backward run of one ESTBUR component.
#[derive(Clone, Copy, Default)]
struct Directions {
    forward: f64,
    backward: f64,
}

/// One index of ESTBUR's six filtered series: trend, seasonal and cycle,
/// each run forward and backward.
#[derive(Clone, Copy, Default)]
struct Components {
    trend: Directions,
    seasonal: Directions,
    cycle: Directions,
}

impl Components {
    /// ESTBUR's six right-hand-side columns, in its order: forward and
    /// backward trend, seasonal, then cycle.
    fn columns(self) -> [f64; 6] {
        [
            self.trend.forward,
            self.trend.backward,
            self.seasonal.forward,
            self.seasonal.backward,
            self.cycle.forward,
            self.cycle.backward,
        ]
    }

    fn from_columns(&[ft, bt, fs, bs, fc, bc]: &[f64; 6]) -> Self {
        Self {
            trend: Directions {
                forward: ft,
                backward: bt,
            },
            seasonal: Directions {
                forward: fs,
                backward: bs,
            },
            cycle: Directions {
                forward: fc,
                backward: bc,
            },
        }
    }
}

/// ESTBUR's filtered series (ansub3.f:183–198): each partition filter run
/// over both extended directions, in six separately ordered accumulators.
fn filtered_sums(
    gains: &[Gains],
    forward: &[f64],
    backward: &[f64],
    filtered_len: usize,
) -> Result<Vec<Components>, Error> {
    (0..filtered_len)
        .map(|i| {
            let sum = |stage, total: f64, gain: f64, value: f64| {
                finite(gain.mul_add(value, total), stage, Some(i))
            };
            let mut sums = Components::default();
            for (j, gain) in gains.iter().enumerate() {
                let (f, b) = (forward[i + j], backward[i + j]);
                let Components {
                    trend,
                    seasonal,
                    cycle,
                } = &mut sums;
                trend.forward = sum(Quantity::ForwardTrendSum, trend.forward, gain.trend, f)?;
                trend.backward = sum(Quantity::BackwardTrendSum, trend.backward, gain.trend, b)?;
                seasonal.forward = sum(
                    Quantity::ForwardSeasonalSum,
                    seasonal.forward,
                    gain.seasonal,
                    f,
                )?;
                seasonal.backward = sum(
                    Quantity::BackwardSeasonalSum,
                    seasonal.backward,
                    gain.seasonal,
                    b,
                )?;
                cycle.forward = sum(Quantity::ForwardCycleSum, cycle.forward, gain.cycle, f)?;
                cycle.backward = sum(Quantity::BackwardCycleSum, cycle.backward, gain.cycle, b)?;
            }
            Ok(sums)
        })
        .collect()
}

/// ESTBUR's endpoint equations (ansub3.f:215–318), 2p-2 rows and six RHS,
/// then its reverse traversal (ansub3.f:329–359): six backward recurrences
/// through the MA from the solved endpoint values.
fn endpoint_recurrence(
    model: ExtractionModel<'_>,
    y: &[Components],
    ma: &[f64],
    totden: &[f64],
    zaf: f64,
    zab: f64,
) -> Result<Vec<Components>, Error> {
    let (p, q) = (totden.len(), ma.len());
    let endpoint_len = p + q - 2;
    // y holds n+q-1 filtered values; the recurrence covers the first n+q-p.
    let recurrence_len = y.len() + 1 - p;
    // Row r pairs with filtered index recurrence_len + endpoint_len - 1 - r.
    let columns = endpoint_len + 6;
    let mut endpoint = vec![0.0; endpoint_len * columns];
    for (r, row) in endpoint.chunks_exact_mut(columns).enumerate() {
        if r + 1 < q {
            row[r..r + p].copy_from_slice(totden);
            row[endpoint_len] = 0.5 * zaf;
            row[endpoint_len + 1] = 0.5 * zab;
        } else {
            for (cell, &coefficient) in row[r + 1 - q..=r].iter_mut().zip(ma.iter().rev()) {
                *cell = coefficient;
            }
            row[endpoint_len..]
                .copy_from_slice(&y[recurrence_len + endpoint_len - 1 - r].columns());
        }
    }
    let endpoint = RowMajorMatrix::new(endpoint, columns);
    let endpoint = match model.fixed_size_family() {
        Some(family) => solve_estbur(model.frequency(), EstburSystem::Endpoint(family), endpoint),
        None => solve(endpoint, endpoint_len, 6),
    }
    .map_err(|failure| failure.at(Stage::Extraction))?;
    let mut x = vec![Components::default(); recurrence_len + endpoint_len];
    for (index, (slot, row)) in x[recurrence_len..]
        .iter_mut()
        .rev()
        .zip(endpoint.rows())
        .enumerate()
    {
        let sides = row
            .get(endpoint_len..)
            .and_then(|sides| <&[f64; 6]>::try_from(sides).ok())
            .ok_or_else(|| {
                Failure::Unusable {
                    quantity: "linear system column count",
                    index: Some(index),
                    value: row.len() as f64,
                }
                .at(Stage::Extraction)
            })?;
        *slot = Components::from_columns(sides);
    }

    for m in (0..recurrence_len).rev() {
        let mut sums = y[m].columns();
        for (j, &coefficient) in ma.iter().enumerate().skip(1) {
            let later = x[m + j].columns();
            for (component, (sum, value)) in sums.iter_mut().zip(later).enumerate() {
                *sum = finite(
                    (-coefficient).mul_add(value, *sum),
                    Quantity::BackwardRecurrence,
                    Some(m * 6 + component),
                )?;
            }
        }
        x[m] = Components::from_columns(&sums);
    }
    Ok(x)
}

fn finite(value: f64, quantity: Quantity, index: Option<usize>) -> Result<f64, Error> {
    super::failure::usable(quantity.label(), index, value)
        .map_err(|failure| failure.at(Stage::Extraction))
}

#[cfg(test)]
mod tests;
