//! Part D (`x11pt3.f`): the final seasonal, trend-cycle, adjusted series and
//! irregular, and how a run ends.

use super::extreme_weights::replace_extremes;
use super::mode::{remove, remove_checked};
use super::seasonal_factors::{SeasonalSmoothing, SeasonalSmoothingRequest, seasonal_filter};
use super::trend::{TrendEstimate, replace_nonpositive_trend, trend_error, variable_trend};
use super::{
    FinalCExtremes, Iteration, Pass, Prepass, PrepassState, RunContext, RunPurpose,
    SeasonalityTests,
};
use crate::error::{NumericalDetail, X11PseudoAdditiveStage as PseudoStage};
use crate::x11::Components;
use crate::x11::pseudo_additive as pseudo;
use crate::x11::seasonal::{ResolvedSeasonalFilter, SeasonalFilterSet};
use crate::{Error, Warning, x11};

impl Iteration {
    /// Part D's shared step (`x11pt3.f`): seasonality diagnostics, the MSR
    /// filter choice, and the final seasonal, trend-cycle, adjusted series and
    /// irregular.
    pub(super) fn part_d<T: SeasonalityTests>(
        &mut self,
        context: &RunContext<'_>,
        mut final_si: Vec<f64>,
        final_iteration_trend: &[f64],
        diagnostic_c_extremes: Vec<f64>,
        tests: T,
        state: PrepassState,
    ) -> Result<PartD<T::Diagnostics>, Error> {
        let &RunContext {
            dates,
            mode,
            period,
            trend_only,
            ref observed,
            ..
        } = context;
        // `x11pt3.f` tests a separate, unmodified D8. The final SI gets its C
        // extremes back afterwards.
        let diagnostic_d8 = final_si
            .iter()
            .zip(&diagnostic_c_extremes)
            .map(|(&value, &extreme)| mode.restore(value, extreme))
            .collect::<Vec<_>>();
        let seasonality = tests.complete(context, &diagnostic_d8, &mut self.warnings);
        if let Some(extremes) = &self.final_c_extremes {
            restore_final_si(&mut final_si, extremes, observed.end, period, mode);
        }
        let selection = if trend_only {
            crate::x11::msr::Selection {
                filters: context.filters,
                outcome: None,
            }
        } else if state.consumed_msr {
            crate::x11::msr::Selection {
                filters: context.filters,
                outcome: Some(crate::x11::MsrSelection::SuppressedByEasterEstimation),
            }
        } else {
            crate::x11::msr::select_checked(&final_si, dates, observed.end, mode, &context.filters)?
        };
        let quality_history = state.history.unwrap_or_else(|| {
            crate::x11::quality::QualityFilterHistory::resolve(
                context.filters.as_slice(),
                selection.filters.as_slice(),
                selection.outcome.as_ref(),
            )
        });
        let (seasonal, modified_adjusted) = self.final_seasonal(
            context,
            &final_si,
            final_iteration_trend,
            &selection.filters,
        )?;
        let (trend, final_mode) = self.final_trend_cycle(context, &modified_adjusted, &seasonal)?;
        let (seasonally_adjusted, irregular) =
            self.final_adjustment(context, final_mode, &trend.values, &seasonal)?;
        let mut eligible = if trend_only {
            vec![true; observed.len()]
        } else {
            crate::x11::quality::intermediate_eligibility(
                mode,
                &modified_adjusted,
                observed.clone(),
            )
        };
        if let Some(previous) = state.eligible {
            for (current, previous) in eligible.iter_mut().zip(previous) {
                *current &= previous;
            }
        }
        Ok(PartD {
            series: FinalSeries {
                seasonal,
                seasonally_adjusted,
                trend,
                irregular,
            },
            selection,
            quality_history,
            eligible,
            final_si,
            diagnostic_c_extremes,
            diagnostic_d8,
            seasonality,
        })
    }

    /// Components from Part D: the observed span of each final series, the
    /// spectral and quality workspaces, and the run's diagnostics.
    pub(super) fn into_components(
        self,
        context: &RunContext<'_>,
        part_d: PartD<crate::diagnostics::SeasonalityTests>,
    ) -> Components {
        let &RunContext {
            dates,
            mode,
            period,
            trend_only,
            ref observed,
            ..
        } = context;
        let PartD {
            series:
                FinalSeries {
                    seasonal,
                    seasonally_adjusted,
                    trend,
                    irregular,
                },
            selection,
            quality_history,
            eligible,
            final_si,
            diagnostic_c_extremes,
            diagnostic_d8,
            seasonality,
        } = part_d;
        let moving_seasonality_ratio = if trend_only {
            Ok(0.0)
        } else {
            crate::x11::msr::quality_ratio(&final_si[..observed.end], period, mode, dates)
        };
        #[cfg(feature = "__conditioning-scales")]
        let conditioning_scales = if self.log_additive.is_some()
            || !matches!(mode, x11::Mode::Additive | x11::Mode::Multiplicative)
        {
            crate::x11::ConditioningScales::default()
        } else {
            crate::x11::ConditioningScales {
                trend_cycle: trend.operation_scales[observed.clone()].to_vec(),
                irregular: irregular_conditioning_scales(
                    mode == x11::Mode::Multiplicative,
                    &seasonally_adjusted,
                    &trend.values,
                    &irregular,
                    &trend.operation_scales,
                )[observed.clone()]
                .to_vec(),
            }
        };
        #[cfg(not(feature = "__conditioning-scales"))]
        let conditioning_scales = crate::x11::ConditioningScales::default();
        Components {
            spectral_workspace: crate::x11::spectral::SpectralWorkspace {
                // `None` for a pseudo-additive run until its quality
                // completion supplies it.
                adjusted_original: self
                    .modified
                    .iter()
                    .zip(&diagnostic_c_extremes)
                    .enumerate()
                    .map(|(i, (&value, &extreme))| {
                        crate::x11::spectral::original_operand(
                            mode,
                            value,
                            extreme,
                            final_si[i],
                            diagnostic_d8[i],
                        )
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|operands| operands[observed.clone()].to_vec()),
                part_f_trend: None,
                modified_adjusted: None,
                modified_irregular: None,
                level_shift_round_trip: crate::x11::spectral::LevelShiftRoundTrip::Unknown,
            },
            decomposition: context.options.decomposition(),
            mode,
            quality_workspace: crate::x11::quality::QualityWorkspace {
                dates,
                observed: observed.clone(),
                seasonal: seasonal[..observed.end].to_vec(),
                base_trend: trend.values[observed.clone()].to_vec(),
                base_irregular: irregular[observed.clone()].to_vec(),
                intermediate_eligible: eligible,
                irregular_trend_ratio: trend.unscaled_ratio,
                moving_seasonality_ratio,
                history: quality_history,
            },
            seasonal: seasonal[observed.clone()].to_vec(),
            seasonally_adjusted: seasonally_adjusted[observed.clone()].to_vec(),
            trend_cycle: trend.values[observed.clone()].to_vec(),
            irregular: irregular[observed.clone()].to_vec(),
            final_henderson_terms: trend.terms,
            msr_selection: selection.outcome,
            calendar_sigma_selection: self.extreme_scale.selection(),
            seasonality,
            warnings: self.warnings,
            intermediate_extreme_weights: self.intermediate_extreme_weights,
            conditioning_scales,
        }
    }

    /// D10, and D11's operand: the final seasonal from the final SI under
    /// the MSR-resolved filters, shrunk and exponentiated as requested, and
    /// the modified series it adjusts.
    fn final_seasonal(
        &self,
        context: &RunContext<'_>,
        final_si: &[f64],
        final_iteration_trend: &[f64],
        selected: &SeasonalFilterSet<x11::SeasonalFilter>,
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let &RunContext {
            values,
            dates,
            mode,
            options,
            trend_only,
            ref observed,
            ..
        } = context;
        // `sfmsr.f` replaces only the MSR entries. The provisional ones still
        // take 3x5, even beside an entry that selected 3x9.
        let final_filters = selected.resolve(ResolvedSeasonalFilter::S3x5);
        let smoothing = if trend_only {
            SeasonalSmoothing {
                values: context.neutral_seasonal(),
                last_filter: ResolvedSeasonalFilter::S3x5,
            }
        } else {
            seasonal_filter(
                final_si,
                0..values.len(),
                mode,
                SeasonalSmoothingRequest {
                    filters: &final_filters,
                    short_series_filters: options.short_series_filters(),
                },
                dates,
            )?
        };
        let mut seasonal = smoothing.values;
        let modified_adjusted = if trend_only {
            self.modified.clone()
        } else if mode == x11::Mode::PseudoAdditive {
            pseudo::adjusted_series(
                &self.modified,
                final_iteration_trend,
                &seasonal,
                dates,
                PseudoStage::FinalAdjustment,
            )?
        } else {
            remove(mode, &self.modified, &seasonal)
        };
        let shrinkage_context = crate::x11::shrinkage::ShrinkageContext {
            mode,
            dates,
            requested_axis: context.requested_axis,
            observed: observed.clone(),
            last_filter: smoothing.last_filter,
        };
        let shrinkage_tail =
            if !trend_only && options.seasonal_shrinkage() != crate::x11::SeasonalShrinkage::None {
                Some(crate::x11::shrinkage::prepare_tail(
                    &seasonal,
                    &shrinkage_context,
                ))
            } else {
                None
            };
        if let Some(phase) = &self.log_additive {
            phase.exponentiate_seasonal(&mut seasonal)?;
        }
        if let Some(tail) = &shrinkage_tail {
            crate::x11::shrinkage::apply(
                options.seasonal_shrinkage(),
                final_si,
                &mut seasonal,
                tail,
                shrinkage_context,
            )?;
        }
        Ok((seasonal, modified_adjusted))
    }

    /// D12: the final Henderson trend-cycle. A log-additive run finishes it
    /// in levels and continues as multiplicative.
    fn final_trend_cycle(
        &mut self,
        context: &RunContext<'_>,
        modified_adjusted: &[f64],
        seasonal: &[f64],
    ) -> Result<(TrendEstimate, x11::Mode), Error> {
        let &RunContext {
            dates,
            mode,
            options,
            period,
            ref observed,
            ..
        } = context;
        let mut trend = variable_trend(
            modified_adjusted,
            period,
            Pass::D,
            self.trend_ic,
            mode,
            observed.clone(),
            options,
        )
        .map_err(|failure| trend_error(dates, x11::TrendStage::FinalTrendCycle, failure))?;
        self.trend_ic = trend.trend_ic;
        if trend.nonpositive_replacements > 0 {
            self.warnings.push(Warning::NonPositiveTrendReplaced {
                stage: x11::TrendStage::FinalTrendCycle,
                replacements: trend.nonpositive_replacements,
            });
        }
        let Some(phase) = &self.log_additive else {
            return Ok((trend, mode));
        };
        phase.finish_trend(
            &mut trend.values,
            seasonal,
            period,
            options.seven_term_endpoints(),
        )?;
        let replacements = replace_nonpositive_trend(&mut trend.values, observed.clone(), None)
            .map_err(|failure| trend_error(dates, x11::TrendStage::FinalTrendCycle, failure))?;
        if replacements > 0 {
            self.warnings.push(Warning::NonPositiveTrendReplaced {
                stage: x11::TrendStage::FinalTrendCycle,
                replacements,
            });
        }
        Ok((trend, x11::Mode::Multiplicative))
    }

    /// D11 and D13: the seasonally adjusted series and the final irregular.
    /// The legacy Easter final run adjusts the raw series, not the
    /// Easter-corrected working series.
    fn final_adjustment(
        &self,
        context: &RunContext<'_>,
        final_mode: x11::Mode,
        trend_cycle: &[f64],
        seasonal: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let &RunContext {
            values,
            raw,
            dates,
            mode,
            trend_only,
            purpose,
            ..
        } = context;
        let final_original = if purpose == RunPurpose::LegacyEasterFinal {
            raw
        } else {
            &self.original
        };
        let seasonally_adjusted = if trend_only {
            final_original.to_vec()
        } else if mode == x11::Mode::PseudoAdditive {
            pseudo::adjusted_series(
                final_original,
                trend_cycle,
                seasonal,
                dates,
                PseudoStage::FinalAdjustment,
            )?
        } else {
            remove(
                final_mode,
                if self.log_additive.is_some() {
                    values
                } else {
                    final_original
                },
                seasonal,
            )
        };
        let irregular = remove_checked(
            final_mode,
            &seasonally_adjusted,
            trend_cycle,
            0..values.len(),
            dates,
            PseudoStage::FinalAdjustment,
        )?;
        Ok((seasonally_adjusted, irregular))
    }

    /// The legacy Easter prepass keeps its irregular, its diagnostics and
    /// what the final run resumes from, not components.
    pub(super) fn into_preliminary(
        self,
        context: &RunContext<'_>,
        part_d: PartD<()>,
    ) -> Result<Prepass, Error> {
        let &RunContext {
            raw,
            axis,
            mode,
            trend_only,
            ref observed,
            ..
        } = context;
        let PartD {
            series,
            selection,
            quality_history,
            mut eligible,
            ..
        } = part_d;
        for (component, values) in [
            (crate::error::Component::Seasonal, &series.seasonal),
            (
                crate::error::Component::SeasonallyAdjusted,
                &series.seasonally_adjusted,
            ),
            (crate::error::Component::TrendCycle, &series.trend.values),
            (crate::error::Component::Irregular, &series.irregular),
        ] {
            if let Some(index) = values[observed.clone()]
                .iter()
                .position(|value| !value.is_finite())
            {
                return Err(Error::from(NumericalDetail::NonFiniteOutput {
                    output: crate::error::AdjustmentOutput::Component(component),
                    index,
                }));
            }
        }
        if mode == x11::Mode::PseudoAdditive && !trend_only {
            for (eligible, &value) in eligible
                .iter_mut()
                .zip(&series.seasonally_adjusted[observed.clone()])
            {
                *eligible &= value > 0.0;
            }
        }
        let consumed_msr = matches!(
            selection.outcome,
            Some(
                crate::x11::MsrSelection::Selected { .. }
                    | crate::x11::MsrSelection::ShortHistoryFallback { .. }
            )
        );
        // The prepass makes no seasonality tests and carries no editor
        // warning, so its passes warn only of trend replacements.
        debug_assert!(
            self.warnings
                .iter()
                .all(|warning| matches!(warning, Warning::NonPositiveTrendReplaced { .. }))
        );
        let trend_warnings = self
            .warnings
            .into_iter()
            .filter_map(|warning| match warning {
                Warning::NonPositiveTrendReplaced {
                    stage,
                    replacements,
                } => Some(crate::x11::EasterPreliminaryTrendWarning {
                    stage,
                    replacements,
                }),
                _ => None,
            })
            .collect();
        Ok(Prepass {
            trend_warnings,
            warnings: crate::x11::seasonality::preliminary_easter_warning(
                &series.seasonally_adjusted[observed.clone()],
                raw,
                axis,
                mode,
            ),
            irregular: series.irregular[observed.clone()].to_vec(),
            msr: selection.outcome,
            sigma: self.extreme_scale.selection(),
            state: PrepassState {
                sigma: self.extreme_scale.sigma,
                consumed_msr,
                history: Some(quality_history),
                eligible: Some(eligible),
            },
        })
    }
}

/// What Part D's shared step hands to `into_components` or
/// `into_preliminary`, with `S` the run's seasonality diagnostics.
pub(super) struct PartD<S> {
    series: FinalSeries,
    selection: crate::x11::msr::Selection,
    quality_history: crate::x11::quality::QualityFilterHistory,
    eligible: Vec<bool>,
    /// The final SI after any C-extreme restoration, for components only.
    final_si: Vec<f64>,
    /// C extremes and the unmodified D8, for components only.
    diagnostic_c_extremes: Vec<f64>,
    diagnostic_d8: Vec<f64>,
    seasonality: S,
}

/// Part D's final series on the full extended axis.
struct FinalSeries {
    seasonal: Vec<f64>,
    seasonally_adjusted: Vec<f64>,
    /// The final trend-cycle is `trend.values`.
    trend: TrendEstimate,
    irregular: Vec<f64>,
}

/// `x11pt3.f`: put the C extremes back into the final SI (`addmul.f`), then
/// replace its extremes under the C weights (`replac.f`). Both run through the
/// last observation, backcasts included; the forecasts keep their modified
/// values.
pub(super) fn restore_final_si(
    si: &mut [f64],
    state: &FinalCExtremes,
    observed_end: usize,
    period: usize,
    mode: x11::Mode,
) {
    for (value, &extreme) in si[..observed_end].iter_mut().zip(&state.component) {
        *value = mode.restore(*value, extreme);
    }
    replace_extremes(si, &state.weights, 0..observed_end, period);
}

#[cfg(feature = "__conditioning-scales")]
fn irregular_conditioning_scales(
    multiplicative: bool,
    seasonally_adjusted: &[f64],
    trend_cycle: &[f64],
    irregular: &[f64],
    trend_cycle_scales: &[f64],
) -> Vec<f64> {
    seasonally_adjusted
        .iter()
        .zip(trend_cycle)
        .zip(irregular.iter().zip(trend_cycle_scales))
        .map(
            |((&seasonally_adjusted, &trend_cycle), (&irregular, &trend_cycle_scale))| {
                let propagated = if multiplicative {
                    seasonally_adjusted.abs() / (trend_cycle * trend_cycle).abs()
                        * trend_cycle_scale
                } else {
                    trend_cycle_scale
                };
                propagated.max(irregular.abs()).max(1.0)
            },
        )
        .collect()
}
