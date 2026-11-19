//! The AR(1) family, `(1 1 0)(0 1 1)` and `(1 1 0)(0 0 0)`, on the shared
//! ANALTS run loop.
//!
//! The model is an AR coefficient and, in the first form, a seasonal MA
//! coefficient. A pass computes CALCFX's residuals for the imported
//! coefficients or refits them with SEARCH; the completed pass runs FCAST,
//! SPECTRUM's canonical decomposition with the AR root allocated by F1RST,
//! the ESTBUR extraction and DETCOMP's final components. A root allocated to
//! the transitory component gives the run a transitory series.

pub(crate) mod model;
pub(crate) mod refit;
pub(crate) mod residual;

use super::effects::FittedEffectsInput;
use super::extraction::{ExtractedComponents, FinalWithTransitory};
use super::mean::{MeanCorrection, MeanFamily, RequestedMean, mean_correction};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput,
    SeatsTransform, run_family,
};
use super::retry::Init;
use super::tails::{
    ForwardTailState, MeanForcing, SeriesTails, TailInputs, ar1_bj_ma, bphist_ordered,
    finish_tails_numeric, resolve_horizons,
};
use crate::Frequency;
use crate::seats::failure::{CanonicalComponent, Error, Failure, Stage};
use model::{ResolvedAr1SeasonalMa, ar1_seasonal_ma_geometry};
use refit::{Ar1SeasonalMaRefit, refit_prepared_ar1_seasonal_ma};
use residual::ar1_residuals;

/// A completed AR(1) run.
pub(crate) type Ar1Run = Composed<ResolvedAr1SeasonalMa, Ar1SeasonalMaRefit>;

/// Runs SEATS on the imported AR(1) model, restoring `calendar`'s fitted
/// effects when given.
pub(crate) fn compose_ar1(
    imported: &ResolvedAr1SeasonalMa,
    input: RunInput<'_>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<Ar1Run, Error> {
    run_family(&Ar1(imported), input, calendar)
}

struct Ar1<'a>(&'a ResolvedAr1SeasonalMa);

/// ESTBUR's components, with what the final components read from the
/// canonical decomposition.
struct Ar1Decomposition {
    raw: ExtractedComponents,
    cycle_variance: f64,
    has_seasonal_component: bool,
}

impl Family for Ar1<'_> {
    type Start = ResolvedAr1SeasonalMa;
    type Model = ResolvedAr1SeasonalMa;
    type Refit = Ar1SeasonalMaRefit;
    type Tails = SeriesTails;
    type Decomposition = Ar1Decomposition;
    type Failure = Error;

    const TMU_BEFORE_LOG: bool = false;

    fn frequency(&self) -> Frequency {
        self.0.frequency()
    }

    fn initial_start(&self) -> Result<ResolvedAr1SeasonalMa, Error> {
        Ok(self.0.clone())
    }

    fn pass_shape(
        &self,
        model: &ResolvedAr1SeasonalMa,
        _observed_len: usize,
        _requested: RequestedMean,
    ) -> Result<PassShape, Error> {
        Ok(PassShape {
            differences: model.differences(),
            // SEARCH estimates the AR and, if any, the seasonal MA.
            free_parameters: 1 + usize::from(model.has_seasonal()),
        })
    }

    fn fit_pass(
        &self,
        model: &mut ResolvedAr1SeasonalMa,
        init: Init,
        pass: &DifferencedPass,
        _requested: RequestedMean,
    ) -> Result<FittedPass<ResolvedAr1SeasonalMa, Ar1SeasonalMaRefit>, Error> {
        let frequency = model.frequency();
        let seasonal = model.has_seasonal();
        let (residuals, refit) = if init == Init::Imported2 {
            let residuals = ar1_residuals(frequency, model.parameters(), &pass.wd, seasonal)?;
            (residuals, None)
        } else {
            let refit = refit_prepared_ar1_seasonal_ma(frequency, &pass.wd, &pass.auto, seasonal)?;
            *model = ResolvedAr1SeasonalMa::from_refit(frequency, refit.parameters, seasonal)?;
            (refit.residuals.clone(), Some(refit))
        };
        Ok(FittedPass {
            model: model.clone(),
            residuals,
            refit,
        })
    }

    fn correct_mean(
        &self,
        model: &ResolvedAr1SeasonalMa,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, Error> {
        let phi = model.parameters()[0];
        let family = if model.has_seasonal() {
            MeanFamily::Ar1(phi)
        } else {
            MeanFamily::NonseasonalAr1(phi)
        };
        mean_correction(a, requested, wm, family)
    }

    fn tails(
        &self,
        model: &mut ResolvedAr1SeasonalMa,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<SeriesTails, Error> {
        finish_ar1_tails(model, series, wd, state)
    }

    fn decompose(
        &self,
        model: &ResolvedAr1SeasonalMa,
        series: &[f64],
        tails: &SeriesTails,
    ) -> Result<Ar1Decomposition, Error> {
        let geometry = ar1_seasonal_ma_geometry(model)?;
        let prefix =
            super::decomposition::rational::lower_degree_rational_prefix(&geometry.inputs)?;
        let minima =
            super::decomposition::minima::spectral_minima_prefix(model.frequency(), prefix);
        if minima.irregular_candidate < 0.0 {
            return Err(Failure::NegativeVariance {
                component: CanonicalComponent::Irregular,
                variance: minima.irregular_candidate,
            }
            .at(Stage::CanonicalDecomposition));
        }
        let canonical = super::decomposition::canonical::prepare_canonical(minima);
        let has_seasonal_component = geometry.inputs.seasonal.combined.len() > 1;
        let adjusted = super::decomposition::adjusted::factor_adjusted(
            canonical,
            geometry.inputs,
            super::trading_day::EffectiveTradingDay::Ordinary,
        )?;
        let cycle_variance = adjusted.cycle.as_ref().map_or(0.0, |cycle| cycle.variance);
        if cycle_variance < 0.0 {
            return Err(Failure::NegativeVariance {
                component: CanonicalComponent::Cycle,
                variance: cycle_variance,
            }
            .at(Stage::CanonicalDecomposition));
        }
        super::decomposition::adjusted::check_show_comp(&adjusted)?;
        let raw = super::extraction::extract_ar1(
            model,
            &adjusted.inputs,
            &adjusted.factors.preparation,
            series,
            tails,
        )?;
        Ok(Ar1Decomposition {
            raw,
            cycle_variance,
            has_seasonal_component,
        })
    }

    fn final_components(
        &self,
        decomposition: Ar1Decomposition,
        restoration: Restoration<'_>,
        transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Error> {
        let Ar1Decomposition {
            raw,
            cycle_variance,
            has_seasonal_component,
        } = decomposition;
        let mut saved = match restoration {
            Restoration::FittedEffects(calendar) => {
                super::effects::final_with_effects_and_transitory(calendar, &raw, cycle_variance)?
            }
            Restoration::Log(log) => {
                super::log_transform::final_log_with_transitory(log, &raw, cycle_variance)?
            }
            Restoration::Identity(original) => {
                super::extraction::final_identity_with_transitory(original, &raw, cycle_variance)?
            }
        };
        if !has_seasonal_component {
            restoration.without_seasonal_roots(&mut saved.saved, transform);
        }
        Ok(saved)
    }
}

/// FCAST for AR(1) with an optional seasonal MA; the AR coefficient forces
/// the mean.
pub(super) fn finish_ar1_tails(
    model: &ResolvedAr1SeasonalMa,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
) -> Result<SeriesTails, Error> {
    let frequency = model.frequency();
    let s = usize::from(frequency.periods_per_year());
    let parameters = model.parameters();
    let bj_ma = ar1_bj_ma(model);
    let mut horizons = resolve_horizons(frequency, observed.len());
    horizons.fhi = horizons.fh.max(2 * s);
    let inputs = TailInputs {
        bj_ma,
        horizons,
        bphist: bphist_ordered(s, &[1.0, -parameters[0]], model.differences()),
        forcing_ar: MeanForcing::One(parameters[0]),
        order: model.differences(),
        working_horizon: horizons.fhi,
    };
    let backward_ma = inputs.bj_ma.clone();
    finish_tails_numeric(frequency, observed, wd, state, inputs, |back| {
        ar1_residuals(frequency, parameters, back, model.has_seasonal())
            .map(|residuals| (residuals, backward_ma, ()))
            .map_err(|error| error.restage(Stage::Tails))
    })
    .map(|core| core.tails)
}
