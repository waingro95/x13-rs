//! The airline family, `(0 1 1)(0 1 1)`, on the shared ANALTS run loop.
//!
//! The model is two MA coefficients. A pass computes CALCFX's residuals for the
//! imported coefficients or refits both with SEARCH; the completed pass runs
//! FCAST on first differences, the ESTBUR extraction and DETCOMP's final
//! components.

pub(crate) mod model;
pub(crate) mod refit;

use super::effects::FittedEffectsInput;
use super::extraction::{
    ExtractedComponents, FinalWithTransitory, extract_airline, final_identity,
};
use super::log_transform::final_log;
use super::mean::{MeanCorrection, residual_mean_state};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput, run_family,
};
use super::residual::{ResidualResult, resolved_residuals};
use super::tails::{
    CoreTails, DifferenceOrder, ForwardTailState, SeriesTails, TailInputs, airline_bj_ma,
    finish_ma_only_tails, resolve_horizons,
};
use crate::Frequency;
use crate::seats::Error;
use model::ResolvedAirline;
use refit::{AirlineRefitResult, refit_prepared_airline};

// The other families and the engine import these names from this module.
pub(crate) use super::mean::RequestedMean;
pub(crate) use super::pipeline::SeatsTransform;
pub(crate) use super::retry::{Init, RetryDecision};

/// A completed airline run.
pub(crate) type AirlineRun = Composed<ResolvedAirline, AirlineRefitResult>;

/// Runs SEATS on the imported airline model, restoring `calendar`'s fitted
/// effects when given.
pub(crate) fn compose_airline(
    imported: &ResolvedAirline,
    input: RunInput<'_>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<AirlineRun, Error> {
    run_family(&Airline(imported), input, calendar)
}

/// The airline's pass: first differences and two MA coefficients.
const PASS_SHAPE: PassShape = PassShape {
    differences: DifferenceOrder::First,
    free_parameters: 2,
};

struct Airline<'a>(&'a ResolvedAirline);

impl Family for Airline<'_> {
    type Start = ResolvedAirline;
    type Model = ResolvedAirline;
    type Refit = AirlineRefitResult;
    type Tails = SeriesTails;
    type Decomposition = ExtractedComponents;
    type Failure = Error;

    const TMU_BEFORE_LOG: bool = true;

    fn frequency(&self) -> Frequency {
        self.0.frequency()
    }

    fn initial_start(&self) -> Result<ResolvedAirline, Error> {
        Ok(self.0.clone())
    }

    fn pass_shape(
        &self,
        _model: &ResolvedAirline,
        _observed_len: usize,
        _requested: RequestedMean,
    ) -> Result<PassShape, Error> {
        Ok(PASS_SHAPE)
    }

    fn fit_pass(
        &self,
        model: &mut ResolvedAirline,
        init: Init,
        pass: &DifferencedPass,
        _requested: RequestedMean,
    ) -> Result<FittedPass<ResolvedAirline, AirlineRefitResult>, Error> {
        let (residuals, refit) = if init == Init::Imported2 {
            (resolved_residuals(model, &pass.wd)?, None)
        } else {
            let search = refit_prepared_airline(model.frequency, &pass.wd, &pass.auto)?;
            // ANALTS's rescale of the parameter standard errors is not run:
            // the components read only the coefficients and the residuals.
            model.ma = search.ma;
            let residuals = ResidualResult {
                scaled_residuals: search.scaled_residuals.clone(),
                determinant_scale: search.determinant_scale,
                objective: search.objective,
                normalized_residuals: search
                    .scaled_residuals
                    .iter()
                    .map(|a| a / search.determinant_scale)
                    .collect(),
            };
            (residuals, Some(search))
        };
        Ok(FittedPass {
            model: model.clone(),
            residuals,
            refit,
        })
    }

    fn correct_mean(
        &self,
        _model: &ResolvedAirline,
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
        model: &mut ResolvedAirline,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<SeriesTails, Error> {
        Ok(finish_tails(model, series, wd, state)?.tails)
    }

    fn decompose(
        &self,
        model: &ResolvedAirline,
        series: &[f64],
        tails: &SeriesTails,
    ) -> Result<ExtractedComponents, Error> {
        extract_airline(model, series, tails)
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

/// FCAST for the airline: its BJ MA, unit AR and first differences.
pub(super) fn finish_tails(
    model: &ResolvedAirline,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
) -> Result<CoreTails, Error> {
    let period = usize::from(model.frequency().periods_per_year());
    let horizons = resolve_horizons(model.frequency(), observed.len());
    let inputs = TailInputs::ma_only(
        period,
        airline_bj_ma(period, model.ma_coefficients()),
        horizons,
        DifferenceOrder::First,
    );
    finish_ma_only_tails(
        model.frequency(),
        observed,
        wd,
        state,
        inputs,
        |backward_wd| resolved_residuals(model, backward_wd),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::airline::refit::RefitCovariance;
    use crate::seats::autocorrelation::AutoLagState;
    use crate::seats::pipeline::{SummaryQ, difference_pass};
    use crate::seats::retry::RetryState;

    fn input(original: &[f64]) -> RunInput<'_> {
        RunInput {
            original,
            transformed: original,
            transform: SeatsTransform::Identity,
            requested_mean: RequestedMean::No,
            summary_q: SummaryQ::Absent,
        }
    }

    fn model(frequency: Frequency) -> ResolvedAirline {
        ResolvedAirline {
            frequency,
            ma: [0.4, 0.6],
        }
    }

    fn prepare(
        original: &[f64],
        frequency: Frequency,
        requested: RequestedMean,
        lags: &mut AutoLagState,
    ) -> DifferencedPass {
        difference_pass(original, frequency, requested, &PASS_SHAPE, lags).unwrap()
    }

    #[test]
    fn retry_mean_sets_the_auto_lags_of_each_pass() {
        // Nine quarterly observations are fewer than a run accepts; they make
        // IQ=2 fall below 2+Imean only while the mean is on.
        let original = (0..9)
            .map(|i| ((i * 37 + 11) % 101) as f64 / 16.0)
            .collect::<Vec<_>>();
        let frequency = Frequency::Quarterly;
        let mut state = RetryState::new(RequestedMean::Yes);
        let mut lags = AutoLagState::new(frequency);
        assert_eq!(state.current_mean(), RequestedMean::Yes);
        let first = prepare(&original, frequency, state.current_mean(), &mut lags);
        assert_eq!((lags.m, lags.iq), (4, 4));
        assert_eq!(
            state.decide(60.0, 0.0, 0.0),
            RetryDecision::RemoveMeanAndRepeat
        );
        assert_eq!(state.current_mean(), RequestedMean::No);
        let second = prepare(&original, frequency, state.current_mean(), &mut lags);
        assert_eq!((lags.m, lags.iq), (4, 2));
        assert_eq!(second.wm, 0.0);
        assert_eq!(
            state.decide(60.0, 0.0, 0.0),
            RetryDecision::ResetToRefit {
                restored_mean: true
            }
        );
        assert_eq!(state.current_mean(), RequestedMean::Yes);
        let third = prepare(&original, frequency, state.current_mean(), &mut lags);
        assert_eq!((lags.m, lags.iq), (4, 4));
        assert_eq!(third.auto, first.auto);
        assert_eq!(first.auto.active_lags, 4);
    }

    #[test]
    fn shortest_series_use_every_differenced_value_as_a_lag() {
        for (frequency, n, m) in [(Frequency::Monthly, 36, 23), (Frequency::Quarterly, 16, 11)] {
            let original = (0..n)
                .map(|i| ((i * 37 + 11) % 101) as f64 / 16.0)
                .collect::<Vec<_>>();
            let mut lags = AutoLagState::new(frequency);
            let prepared = prepare(&original, frequency, RequestedMean::No, &mut lags);
            assert_eq!(prepared.wd.len(), m);
            assert_eq!(prepared.auto.active_lags, m);
        }
    }

    #[test]
    fn refit_at_both_bounds_completes_without_a_covariance() {
        let pattern = [-8, -5, 0, 4, 9, 12, 10, 6, 1, -3, -7, -9];
        let original = (0..180)
            .map(|t| {
                let i = t as i32;
                f64::from(8192 + 4 * i + 8 * pattern[t % 12] + ((37 * i + 11) % 101) - 50) / 64.0
            })
            .collect::<Vec<_>>();
        let prepared = prepare(
            &original,
            Frequency::Monthly,
            RequestedMean::No,
            &mut AutoLagState::new(Frequency::Monthly),
        );
        assert_eq!(prepared.wd.len(), 167);
        let lower =
            refit_prepared_airline(Frequency::Monthly, &prepared.wd, &prepared.auto).unwrap();
        assert_eq!(lower.bound_mask, [true, true]);
        assert_eq!(lower.covariance, RefitCovariance::UnavailableAllFixed);
        // CALCFX returns Na=Nw+Qstar=167+13, not the WD input length.
        assert_eq!(lower.scaled_residuals.len(), 180);
        assert!(lower.scaled_residuals.iter().all(|value| value.is_finite()));
        assert!(lower.objective.is_finite() && lower.objective > 0.0);
        assert!(lower.determinant_scale.is_finite() && lower.determinant_scale > 0.0);
        let result = compose_airline(&model(Frequency::Monthly), input(&original), None)
            .expect("the run needs no covariance");
        let Some(search) = &result.passes[1].refit else {
            panic!("the second pass refits");
        };
        assert_eq!(search.covariance, RefitCovariance::UnavailableAllFixed);
        assert_eq!(search.bound_mask, lower.bound_mask);
        assert_eq!(search.ma, lower.ma);
        assert_eq!(search.scaled_residuals, lower.scaled_residuals);
        assert_eq!(result.passes[1].decision, RetryDecision::Complete);
    }
}
