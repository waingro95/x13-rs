//! The steps of the B, C and D passes before Part D.

use std::ops::Range;

use super::extreme_scale::{ExtremeScaleState, SigmaState};
use super::extreme_weights::{calendar_extreme_weights, extreme_components};
use super::mode::{pseudo_modified, pseudo_trend_modified, remove, remove_checked};
use super::seasonal_factors::{
    SeasonalWeightRequest, extend_seasonal, replace_extremes_and_filter, seasonal_filter,
};
use super::trend::{
    TrendEstimate, centered_period_trend, initial_trend_ic, trend_error, variable_trend,
};
use super::{FinalCExtremes, Iteration, Pass, RunContext, RunPurpose};
use crate::error::{X11ExtremeStage, X11PseudoAdditiveStage as PseudoStage};
use crate::x11::pseudo_additive as pseudo;
use crate::{Error, Warning, x11};

impl Iteration {
    /// The state before pass B. A legacy Easter run with an automatic extreme
    /// scale enters `entsch.f`'s states at `sigma`.
    pub(super) fn new(
        context: &RunContext<'_>,
        sigma: Option<SigmaState>,
        warning: Option<Warning>,
    ) -> Result<Self, Error> {
        let mut extreme_scale = ExtremeScaleState::new(context.options);
        if context.purpose != RunPurpose::Ordinary
            && context.options.extreme_scale() == crate::x11::ExtremeScale::Automatic
        {
            extreme_scale = extreme_scale.with_sigma_entry(sigma.unwrap_or_default());
        }
        let log_additive = if context.mode == x11::Mode::LogAdditive {
            Some(crate::x11::log_additive::Phase::new(
                context.values,
                context.dates,
            )?)
        } else {
            None
        };
        let original = log_additive
            .as_ref()
            .map_or_else(|| context.values.to_vec(), |phase| phase.logged().to_vec());
        Ok(Self {
            log_additive,
            modified: original.clone(),
            original,
            extreme_scale,
            trend_ic: initial_trend_ic(context.period, context.options.trend_filter()),
            warnings: warning.into_iter().collect(),
            intermediate_extreme_weights: Vec::new(),
            final_c_extremes: None,
        })
    }

    /// Every pass's first step: the SI against a centered moving average.
    pub(super) fn initial_si(&self, context: &RunContext<'_>) -> Result<InitialSi, Error> {
        let &RunContext {
            dates,
            mode,
            period,
            ..
        } = context;
        let (trend, span) = centered_period_trend(&self.modified, period);
        let si = remove_checked(
            mode,
            &self.modified,
            &trend,
            span.clone(),
            dates,
            PseudoStage::SeasonalIrregular,
        )?;
        Ok(InitialSi { si, trend, span })
    }

    /// The trend half of every pass, from its initial SI: a seasonal estimate
    /// (with extreme replacement on B), the Henderson trend of the seasonally
    /// adjusted series, and the SI against it.
    pub(super) fn estimate_trend(
        &mut self,
        context: &RunContext<'_>,
        pass: Pass,
        initial: InitialSi,
    ) -> Result<(Vec<f64>, TrendEstimate), Error> {
        let &RunContext {
            values,
            dates,
            mode,
            options,
            period,
            trend_only,
            ref observed,
            ..
        } = context;
        let InitialSi {
            mut si,
            trend: initial_trend,
            span: central_span,
        } = initial;
        let mut seasonal = if pass == Pass::B {
            replace_extremes_and_filter(
                &mut si,
                context.extreme_spans(central_span.clone(), period / 2),
                mode,
                &mut self.extreme_scale,
                SeasonalWeightRequest {
                    dates,
                    smoothing: context.first_smoothing(),
                    robust: context.robust_at(X11ExtremeStage::InitialSeasonalIrregular),
                },
            )?
        } else if trend_only {
            context.neutral_seasonal()
        } else {
            seasonal_filter(
                &si,
                central_span.clone(),
                mode,
                context.first_smoothing(),
                dates,
            )?
            .values
        };
        extend_seasonal(&mut seasonal, central_span.clone(), period);

        let adjusted = if trend_only {
            self.modified.clone()
        } else if mode == x11::Mode::PseudoAdditive {
            pseudo::initial_adjusted(
                &self.modified,
                &initial_trend,
                &seasonal,
                central_span,
                dates,
            )?
        } else {
            remove(mode, &self.modified, &seasonal)
        };
        let trend = variable_trend(
            &adjusted,
            period,
            pass,
            self.trend_ic,
            mode,
            observed.clone(),
            options,
        )
        .map_err(|failure| trend_error(dates, pass.trend_stage(), failure))?;
        self.trend_ic = trend.trend_ic;
        if trend.nonpositive_replacements > 0 {
            self.warnings.push(Warning::NonPositiveTrendReplaced {
                stage: pass.trend_stage(),
                replacements: trend.nonpositive_replacements,
            });
        }
        let si = remove_checked(
            mode,
            &self.modified,
            &trend.values,
            0..values.len(),
            dates,
            PseudoStage::SeasonalIrregular,
        )?;
        Ok((si, trend))
    }

    /// The weighting half of B and C: a refined seasonal (with extreme
    /// replacement on B), the irregular and its extreme weights.
    pub(super) fn weight_extremes(
        &mut self,
        context: &RunContext<'_>,
        pass: Pass,
        mut si: Vec<f64>,
        trend: &[f64],
    ) -> Result<WeightedIrregular, Error> {
        let &RunContext {
            values,
            dates,
            mode,
            options,
            period,
            start_subperiod,
            trend_only,
            ref observed,
            ..
        } = context;
        let seasonal = if pass == Pass::B {
            replace_extremes_and_filter(
                &mut si,
                context.extreme_spans(0..values.len(), 0),
                mode,
                &mut self.extreme_scale,
                SeasonalWeightRequest {
                    dates,
                    smoothing: context.second_smoothing(),
                    robust: context.robust_at(X11ExtremeStage::RefinedSeasonalIrregular),
                },
            )?
        } else if trend_only {
            context.neutral_seasonal()
        } else {
            seasonal_filter(
                &si,
                0..values.len(),
                mode,
                context.second_smoothing(),
                dates,
            )?
            .values
        };
        let adjusted = if trend_only {
            self.original.clone()
        } else if mode == x11::Mode::PseudoAdditive {
            pseudo::adjusted_series(
                &self.original,
                trend,
                &seasonal,
                dates,
                PseudoStage::Irregular,
            )?
        } else {
            remove(mode, &self.original, &seasonal)
        };
        let irregular = remove_checked(
            mode,
            &adjusted,
            trend,
            0..values.len(),
            dates,
            PseudoStage::Irregular,
        )?;
        if pass == Pass::C
            && let Some(phase) = &mut self.log_additive
        {
            phase.retain_c13(&irregular);
        }
        if pass == Pass::B {
            // `x11pt2.f` makes the second test on the backcasts and
            // observations, whatever span the weights are estimated over.
            self.extreme_scale
                .resolve(&irregular, 0..observed.end, period, mode, false, dates)?;
        }
        let stage = if pass == Pass::B {
            X11ExtremeStage::PreliminaryIrregular
        } else {
            X11ExtremeStage::IntermediateIrregular
        };
        let weights = calendar_extreme_weights(
            &irregular,
            context.extreme_spans(0..values.len(), 0),
            period,
            start_subperiod,
            mode,
            options.sigma_limits(),
            self.extreme_scale
                .weight_policy()
                .with_context(context.robust_at(stage)),
        )?;
        if pass == Pass::C {
            self.intermediate_extreme_weights = weights[observed.clone()].to_vec();
        }
        Ok(WeightedIrregular {
            seasonal,
            irregular,
            weights,
        })
    }

    /// The next modified series: the original less the pass's extremes.
    pub(super) fn remove_extremes(
        &mut self,
        context: &RunContext<'_>,
        pass: Pass,
        trend: &[f64],
        weighted: WeightedIrregular,
    ) -> Result<(), Error> {
        let &RunContext {
            dates,
            mode,
            options,
            trend_only,
            ..
        } = context;
        let WeightedIrregular {
            seasonal,
            irregular,
            weights,
        } = weighted;
        let retains_c_extremes = pass == Pass::C
            && (self
                .extreme_scale
                .sigma
                .is_some_and(SigmaState::retains_c_extremes)
                || matches!(
                    options.extreme_scale(),
                    crate::x11::ExtremeScale::ByCalendarPeriod
                        | crate::x11::ExtremeScale::Grouped(_)
                ));
        if retains_c_extremes || mode == x11::Mode::PseudoAdditive {
            // `x11pt2.f` computes `Stex` directly. The final SI is rebuilt
            // from these rounded values and the C weights.
            let component: Vec<_> = extreme_components(mode, &irregular, &weights).collect();
            self.modified = if mode != x11::Mode::PseudoAdditive {
                remove(mode, &self.original, &component)
            } else if trend_only {
                pseudo_trend_modified(trend, &irregular, &component, dates)?
            } else {
                pseudo_modified(trend, &seasonal, &irregular, &component, dates)?
            };
            if retains_c_extremes {
                self.final_c_extremes = Some(FinalCExtremes { component, weights });
            }
        } else {
            self.modified = self
                .original
                .iter()
                .zip(extreme_components(mode, &irregular, &weights))
                .map(|(&original, extreme)| mode.remove(original, extreme))
                .collect();
        }
        Ok(())
    }
}

/// A pass's centered SI, with the trend and span it was taken over.
pub(super) struct InitialSi {
    pub(super) si: Vec<f64>,
    trend: Vec<f64>,
    span: Range<usize>,
}

/// A B or C pass's refined seasonal, its irregular and the extreme weights of
/// that irregular.
pub(super) struct WeightedIrregular {
    seasonal: Vec<f64>,
    pub(super) irregular: Vec<f64>,
    pub(super) weights: Vec<f64>,
}
