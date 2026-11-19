//! The second-difference family, `(0 2 1)(0 1 1)`, on the shared ANALTS run
//! loop.
//!
//! The model is the airline's two MA coefficients on a second regular
//! difference. A pass computes CALCFX's residuals for the imported
//! coefficients or refits both with the airline's SEARCH; the completed pass
//! runs FCAST on second differences, SPECTRUM's canonical decomposition, the
//! ESTBUR extraction and DETCOMP's final components.

pub(crate) mod model;

use super::airline::refit::{AirlineRefitResult, refit_prepared_airline};
use super::decomposition::adjusted::{check_show_comp, factor_adjusted};
use super::decomposition::canonical::prepare_canonical;
use super::decomposition::minima::spectral_minima_prefix;
use super::decomposition::rational::lower_degree_rational_prefix;
use super::effects::FittedEffectsInput;
use super::extraction::{
    ExtractedComponents, FinalWithTransitory, extract_second_difference, final_identity,
};
use super::log_transform::final_log;
use super::mean::{MeanCorrection, RequestedMean, residual_mean_state};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput,
    SeatsTransform, run_family,
};
use super::residual::{ResidualResult, ordered_residuals};
use super::retry::Init;
use super::tails::{
    DifferenceOrder, ForwardTailState, SeriesTails, TailInputs, airline_bj_ma,
    finish_ma_only_tails, resolve_horizons,
};
use super::trading_day::EffectiveTradingDay;
use crate::Frequency;
use crate::seats::failure::{CanonicalComponent, Error, Failure, Stage};
use model::{ResolvedSecondDifference, spectral_inputs};

/// A completed second-difference run.
pub(crate) type SecondDifferenceRun = Composed<ResolvedSecondDifference, AirlineRefitResult>;

/// Runs SEATS on the imported second-difference model, restoring `calendar`'s
/// fitted effects when given.
pub(crate) fn compose_second_difference(
    imported: &ResolvedSecondDifference,
    input: RunInput<'_>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<SecondDifferenceRun, Error> {
    run_family(&SecondDifference(imported), input, calendar)
}

/// The second-difference pass: second differences and two MA coefficients.
const PASS_SHAPE: PassShape = PassShape {
    differences: DifferenceOrder::Second,
    free_parameters: 2,
};

struct SecondDifference<'a>(&'a ResolvedSecondDifference);

impl Family for SecondDifference<'_> {
    type Start = ResolvedSecondDifference;
    type Model = ResolvedSecondDifference;
    type Refit = AirlineRefitResult;
    type Tails = SeriesTails;
    type Decomposition = ExtractedComponents;
    type Failure = Error;

    const TMU_BEFORE_LOG: bool = false;

    fn frequency(&self) -> Frequency {
        self.0.frequency
    }

    fn initial_start(&self) -> Result<ResolvedSecondDifference, Error> {
        Ok(self.0.clone())
    }

    fn pass_shape(
        &self,
        _model: &ResolvedSecondDifference,
        _observed_len: usize,
        _requested: RequestedMean,
    ) -> Result<PassShape, Error> {
        Ok(PASS_SHAPE)
    }

    fn fit_pass(
        &self,
        model: &mut ResolvedSecondDifference,
        init: Init,
        pass: &DifferencedPass,
        _requested: RequestedMean,
    ) -> Result<FittedPass<ResolvedSecondDifference, AirlineRefitResult>, Error> {
        let (residuals, refit) = if init == Init::Refit0 {
            let search = refit_prepared_airline(model.frequency, &pass.wd, &pass.auto)?;
            model.parameters = search.ma;
            let residuals = ResidualResult {
                scaled_residuals: search.scaled_residuals.clone(),
                determinant_scale: search.determinant_scale,
                objective: search.objective,
                normalized_residuals: search
                    .scaled_residuals
                    .iter()
                    .map(|v| v / search.determinant_scale)
                    .collect(),
            };
            (residuals, Some(search))
        } else {
            let period = usize::from(model.frequency.periods_per_year());
            (ordered_residuals(period, model.parameters, &pass.wd)?, None)
        };
        Ok(FittedPass {
            model: model.clone(),
            residuals,
            refit,
        })
    }

    fn correct_mean(
        &self,
        _model: &ResolvedSecondDifference,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, Error> {
        Ok(MeanCorrection {
            statistics: residual_mean_state(a, requested)?,
            wm,
        })
    }

    fn tails(
        &self,
        model: &mut ResolvedSecondDifference,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<SeriesTails, Error> {
        finish_second_difference_tails(model, series, wd, state)
    }

    fn decompose(
        &self,
        model: &ResolvedSecondDifference,
        series: &[f64],
        tails: &SeriesTails,
    ) -> Result<ExtractedComponents, Error> {
        let inputs = spectral_inputs(model);
        let prefix = lower_degree_rational_prefix(&inputs)?;
        let minima = spectral_minima_prefix(model.frequency, prefix);
        if minima.irregular_candidate < 0.0 {
            // gtinpt defaults Lnoadm=false, so SPECTRUM's NOADMISS=0 branch
            // rejects a negative irregular minimum.
            return Err(Failure::NegativeVariance {
                component: CanonicalComponent::Irregular,
                variance: minima.irregular_candidate,
            }
            .at(Stage::CanonicalDecomposition));
        }
        let adjusted = factor_adjusted(
            prepare_canonical(minima),
            inputs,
            EffectiveTradingDay::Ordinary,
        )?;
        check_show_comp(&adjusted)?;
        extract_second_difference(
            model,
            &adjusted.inputs,
            &adjusted.factors.preparation,
            series,
            tails,
        )
    }

    fn final_components(
        &self,
        raw: ExtractedComponents,
        restoration: Restoration<'_>,
        _transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Error> {
        let saved = match restoration {
            Restoration::FittedEffects(calendar) => {
                super::effects::final_with_effects(calendar, &raw)?
            }
            Restoration::Log(log) => final_log(log, &raw)?,
            Restoration::Identity(original) => final_identity(original, &raw)?,
        };
        Ok(FinalWithTransitory {
            saved,
            transitory: None,
        })
    }
}

/// FCAST for the second difference: the airline's BJ MA on second
/// differences, with a working horizon that covers lag 2s+3.
fn finish_second_difference_tails(
    model: &ResolvedSecondDifference,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
) -> Result<SeriesTails, Error> {
    let frequency = model.frequency;
    let s = usize::from(frequency.periods_per_year());
    let ma = model.parameters;
    let horizons = resolve_horizons(frequency, observed.len());
    let inputs = TailInputs {
        working_horizon: horizons.fhi.max(2 * s + 3),
        ..TailInputs::ma_only(s, airline_bj_ma(s, ma), horizons, DifferenceOrder::Second)
    };
    finish_ma_only_tails(frequency, observed, wd, state, inputs, |back| {
        ordered_residuals(s, ma, back)
    })
    .map(|core| core.tails)
}
