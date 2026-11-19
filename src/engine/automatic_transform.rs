//! Automatic transform selection (`trnaic.f`): fits the model with no
//! transform and with logs, each from its own starting values, and compares
//! their AICC.

use crate::api::ResolvedRegArimaSpec;
use crate::arima::{ArmaStart, TransformCandidateFit, fit_transform_candidate};
use crate::error::{NumericalDetail, TransformSelectionUnavailableReason};
use crate::regarima::{TransformSelection, TransformSelectionResult};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Error, RegArimaSpec, Transform};

/// Which model the transform candidates fit.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CandidateModelContext {
    /// The requested ARIMA model, with its starting values.
    ExplicitModel,
    /// `automd.f`'s fresh airline model, installed after `trnaic.f`.
    AutomaticModel,
}

/// The full input series. `editor.f` checks it, and `trnaic.f` builds its
/// length-of-period prior on it, before the response narrows to the model span.
#[derive(Clone, Copy)]
pub(super) struct OriginalSeries<'a> {
    pub(super) values: &'a [f64],
    pub(super) axis: TimeAxis,
}

pub(super) struct ResolvedTransform {
    pub(super) spec: ResolvedRegArimaSpec,
    pub(super) start: ArmaStart,
    pub(super) selection: Option<TransformSelectionResult>,
    pub(super) warnings: Vec<crate::Warning>,
}

/// Everything a transform resolution reads besides the modelspan response:
/// the model the candidates fit, the X-11 decomposition that sets the flow
/// length context, and the full original series.
///
/// Without a model span the response is the original series and the original
/// axis (`TimeAxis::shift(0)` is the identity), so a request with the original
/// series covers that case too.
#[derive(Clone, Copy)]
pub(super) struct TransformRequest<'a> {
    pub(super) context: CandidateModelContext,
    pub(super) decomposition: crate::x11::Decomposition,
    pub(super) original: OriginalSeries<'a>,
}

impl TransformRequest<'_> {
    /// Resolves the transform of `requested` on the modelspan response
    /// `values`; `missing` and `axis` describe that response.
    pub(super) fn resolve(
        self,
        values: &[f64],
        missing: &[usize],
        axis: TimeAxis,
        requested: RegArimaSpec,
    ) -> Result<ResolvedTransform, Error> {
        let requested = match self.context {
            CandidateModelContext::ExplicitModel => requested,
            // `automd.f` installs a fresh airline model after `trnaic.f`, so
            // the candidates fit the airline model. Both entry points validate
            // the request's model policy before they get here.
            CandidateModelContext::AutomaticModel => {
                let airline =
                    ArimaModel::new(ArimaOrder::new(0, 1, 1)?, ArimaOrder::new(0, 1, 1)?)?;
                requested.transform_test_spec(airline)
            }
        };
        let length_context = match self.decomposition {
            crate::x11::Decomposition::SeasonalAdjustment => {
                crate::api::FlowLengthContext::LeapYear
            }
            crate::x11::Decomposition::TrendOnly => crate::api::FlowLengthContext::LengthOfPeriod,
        };
        let options = match requested.transform() {
            TransformSelection::Explicit(_) => {
                return Ok(ResolvedTransform {
                    spec: requested
                        .resolve_explicit()?
                        .with_flow_length_context(length_context),
                    start: ArmaStart::Default,
                    selection: None,
                    warnings: Vec::new(),
                });
            }
            TransformSelection::Automatic(options) => options,
        };
        // `editor.f` skips the test, before any fitting, when the original
        // series has a nonpositive value.
        if let Some(index) = crate::numerics::first_nonpositive(self.original.values) {
            return Ok(ResolvedTransform {
                spec: requested
                    .resolve(Transform::None)?
                    .with_flow_length_context(length_context),
                start: ArmaStart::Default,
                selection: Some(TransformSelectionResult::NonPositiveInput { index }),
                warnings: vec![crate::Warning::AutomaticTransformNonPositiveInput { index }],
            });
        }

        let none = candidate(
            values,
            missing,
            axis,
            requested
                .resolve(Transform::None)?
                .with_flow_length_context(length_context),
            ArmaStart::Default,
            self.original.axis,
            requested.user_regressors(),
        )?;
        let no_transform_aicc = candidate_aicc(&none, Transform::None)?;
        // The log fit starts from the original starting values, not the
        // no-transform estimates: `trnaic.f` restores its Ap1 backup first.
        // With an automatic model, `gtinpt.f` left that backup at zero because
        // no model coefficients existed when `editor.f` ran.
        let log_start = match self.context {
            CandidateModelContext::ExplicitModel => ArmaStart::Default,
            CandidateModelContext::AutomaticModel => ArmaStart::Previous(vec![0.0, 0.0]),
        };
        let log = candidate(
            values,
            missing,
            axis,
            requested
                .resolve(Transform::Log)?
                .with_flow_length_context(length_context),
            log_start,
            self.original.axis,
            requested.user_regressors(),
        )?;
        let log_aicc = candidate_aicc(&log, Transform::Log)?;
        let aic_difference = options.aic_difference();
        let selected = choose(no_transform_aicc, log_aicc, aic_difference);
        let mut warnings = none.warnings;
        warnings.extend(log.warnings);
        Ok(ResolvedTransform {
            // `trnaic.f` restores the estimated leap-year regressor when no
            // transform is selected, whatever length column the trials fitted.
            spec: requested.resolve(selected)?.with_flow_length_context(
                if selected == Transform::None {
                    crate::api::FlowLengthContext::LeapYear
                } else {
                    length_context
                },
            ),
            // `automd.f` restarts the automatic model from the default airline
            // starts. An explicit model starts from the log candidate's
            // estimates.
            start: match self.context {
                CandidateModelContext::ExplicitModel => ArmaStart::Previous(log.parameters),
                CandidateModelContext::AutomaticModel => ArmaStart::Default,
            },
            selection: Some(TransformSelectionResult::Compared {
                selected,
                no_transform_aicc,
                log_aicc,
                aic_difference,
            }),
            warnings,
        })
    }
}

fn choose(no_transform_aicc: f64, log_aicc: f64, aic_difference: f64) -> Transform {
    // `trnaic.f` adds the difference to the log score, then compares; a tie
    // selects no transform.
    if (log_aicc + aic_difference) < no_transform_aicc {
        Transform::Log
    } else {
        Transform::None
    }
}

fn candidate(
    values: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    start: ArmaStart,
    original_axis: TimeAxis,
    user_regressors: &[crate::regarima::UserRegressor],
) -> Result<TransformCandidateFit, Error> {
    // Rebuild canonical columns for each transform: only td/td1coef replace
    // estimated leap year with a fixed prior under log (`rmlnvr.f`).
    let mut regression = RegressionModel::for_transform_candidate(spec, axis, values.len())?;
    regression.add_user_regressors(user_regressors, axis, values.len())?;
    regression.add_missing(missing, axis, values.len())?;
    let adjusted = remove_candidate_fixed_length(values, original_axis, spec, &regression)?;
    // trnaic.f copies the original-origin lomeff to Adj, then prlkhd reads
    // Adj(Adj1st): unlike the response's prefix, this skips any backcast rows.
    let likelihood_prior = if original_axis != axis && !regression.fixed_length().is_empty() {
        Some(regression.evaluate_fixed_length(original_axis, values.len())?)
    } else {
        None
    };
    let mut fit_context = crate::arima::FitContext::unadjusted(values);
    if let Some(prior) = &likelihood_prior {
        fit_context = fit_context.with_likelihood_prior(prior)?;
    }
    fit_transform_candidate(&adjusted, fit_context, axis, spec, regression, start)
}

fn remove_candidate_fixed_length(
    values: &[f64],
    original_axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression: &RegressionModel,
) -> Result<Vec<f64>, Error> {
    if !regression.fixed_length().is_empty() {
        // trnaic.f builds lomeff at Begadj before narrowing to modelspan, then
        // divides by its prefix. The regression model keeps its own prior;
        // `candidate` passes prlkhd its prior separately.
        let prior_axis = original_axis.shift(-i64::from(spec.extensions().backcast_periods()))?;
        let prior = regression.evaluate_fixed_length(prior_axis, values.len())?;
        return Ok(remove_fixed_length(values, &prior));
    }
    Ok(remove_fixed_length(values, regression.fixed_length()))
}

fn candidate_aicc(fit: &TransformCandidateFit, candidate: Transform) -> Result<f64, Error> {
    let reason = if !fit.innovation_variance.is_finite() {
        TransformSelectionUnavailableReason::NonFiniteScore
    } else if fit.innovation_variance == 0.0 {
        TransformSelectionUnavailableReason::ZeroVariance
    } else if let Some(likelihood) = fit.likelihood {
        match likelihood.aicc {
            Some(score) if score.is_finite() => return Ok(score),
            Some(_) => TransformSelectionUnavailableReason::NonFiniteScore,
            None => TransformSelectionUnavailableReason::InsufficientEffectiveObservations,
        }
    } else {
        TransformSelectionUnavailableReason::NonFiniteScore
    };
    Err(Error::from(
        NumericalDetail::TransformSelectionUnavailable { candidate, reason },
    ))
}

/// `values` divided by the fixed length-of-period factors of the same
/// periods, or unchanged when the model has none.
pub(super) fn remove_fixed_length(values: &[f64], fixed_length: &[f64]) -> Vec<f64> {
    if fixed_length.is_empty() {
        return values.to_vec();
    }
    debug_assert_eq!(values.len(), fixed_length.len());
    values
        .iter()
        .zip(fixed_length)
        .map(|(value, factor)| value / factor)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NumericalFailure;
    use crate::regarima::{
        CalendarBasis, CalendarSpec, FlowLengthTreatment, FlowTradingDay, ForecastHorizon,
        ForecastSpec, TradingDayContrast,
    };

    fn airline_with(calendar: CalendarSpec, forecasts: u16, backcasts: u16) -> RegArimaSpec {
        RegArimaSpec::new(ArimaModel::airline())
            .with_calendar(calendar)
            .with_forecast(
                ForecastSpec::new(ForecastHorizon::Periods(forecasts), backcasts).unwrap(),
            )
    }

    fn trading_day(contrast: TradingDayContrast) -> CalendarSpec {
        CalendarSpec::new().with_basis(CalendarBasis::FlowTradingDay(FlowTradingDay::new(contrast)))
    }

    /// `trnaic.f` dates the leap-year prior from the first backcast when it
    /// divides the response, so with a year of backcasts a leap February or
    /// first quarter is divided by the factor of the year before. The
    /// regression model keeps the prior of its own periods.
    #[test]
    fn candidate_response_takes_its_leap_year_prior_from_the_backcast_start() {
        for (axis, backcasts, index, days_a_year_earlier, days, mean) in [
            (
                TimeAxis::monthly(1995, 7).unwrap(),
                12,
                7,
                28.0,
                29.0,
                28.25,
            ),
            (
                TimeAxis::quarterly(1997, 3).unwrap(),
                4,
                10,
                90.0,
                91.0,
                90.25,
            ),
        ] {
            let resolved = airline_with(
                trading_day(TradingDayContrast::WeekdayWeekend),
                0,
                backcasts,
            )
            .resolve(Transform::Log)
            .unwrap();
            let trial = RegressionModel::for_transform_candidate(resolved, axis, 16).unwrap();
            let response =
                remove_candidate_fixed_length(&[100.0; 16], axis, resolved, &trial).unwrap();
            assert_eq!(
                response[index],
                100.0 / (days_a_year_earlier / mean),
                "{axis:?}"
            );
            assert_eq!(trial.fixed_length()[index], days / mean, "{axis:?}");
            assert_eq!(
                trial,
                RegressionModel::from_spec(resolved, axis, 16).unwrap()
            );
        }
    }

    /// The length-of-period prior of a trend-only run is dated the same way:
    /// `trnaic.f` divides by `lomeff` from `Begadj`, the first backcast, while
    /// `prlkhd.f` reads `Adj(Adj1st)`, the observed periods.
    #[test]
    fn candidate_response_takes_its_length_of_period_prior_from_the_backcast_start() {
        let axis = TimeAxis::quarterly(1997, 3).unwrap();
        let resolved = airline_with(trading_day(TradingDayContrast::WeekdayWeekend), 4, 4)
            .resolve(Transform::Log)
            .unwrap()
            .with_flow_length_context(crate::api::FlowLengthContext::LengthOfPeriod);
        let trial = RegressionModel::for_transform_candidate(resolved, axis, 16).unwrap();
        let response = remove_candidate_fixed_length(&[100.0; 16], axis, resolved, &trial).unwrap();
        assert_eq!(response[10], 100.0 / (90.0 / 91.3125));
        assert_eq!(trial.fixed_length()[10], 91.0 / 91.3125);
        assert_eq!(
            trial,
            RegressionModel::from_spec(resolved, axis, 16).unwrap()
        );
    }

    /// Only a log model whose trading day carries a length effect has the
    /// prior. Without backcasts it is that of the response's own periods.
    #[test]
    fn leap_year_prior_needs_logs_and_a_trading_day_length_effect() {
        let axis = TimeAxis::monthly(1995, 7).unwrap();
        let flow = FlowTradingDay::new(TradingDayContrast::SixCoefficient);
        let without_length = flow.with_length_treatment(FlowLengthTreatment::NoLengthEffect);
        for (calendar, transform, backcasts, has_prior) in [
            (
                trading_day(TradingDayContrast::SixCoefficient),
                Transform::Log,
                0,
                true,
            ),
            (
                trading_day(TradingDayContrast::SixCoefficient),
                Transform::None,
                12,
                false,
            ),
            (CalendarSpec::new(), Transform::Log, 12, false),
            (
                CalendarSpec::new().with_basis(CalendarBasis::FlowTradingDay(without_length)),
                Transform::Log,
                12,
                false,
            ),
        ] {
            let resolved = airline_with(calendar, 0, backcasts)
                .resolve(transform)
                .unwrap();
            let trial = RegressionModel::for_transform_candidate(resolved, axis, 16).unwrap();
            assert_eq!(!trial.fixed_length().is_empty(), has_prior);
            let response =
                remove_candidate_fixed_length(&[100.0; 16], axis, resolved, &trial).unwrap();
            let february_1996 = if has_prior {
                100.0 / (29.0 / 28.25)
            } else {
                100.0
            };
            assert_eq!(response[7], february_1996);
        }
    }

    #[test]
    fn log_needs_a_strictly_lower_adjusted_aicc() {
        assert_eq!(choose(10.0, 12.0, -2.0), Transform::None);
        assert_eq!(choose(10.0, 12.0_f64.next_down(), -2.0), Transform::Log);
        assert_eq!(choose(10.0, 12.0_f64.next_up(), -2.0), Transform::None);
        // Subtracting the scores first would choose log after rounding here.
        assert_eq!(choose(1.0, 1.0, -f64::EPSILON / 4.0), Transform::None);
    }

    #[test]
    fn candidate_without_a_finite_aicc_is_an_error() {
        let order = crate::ArimaOrder::new(0, 0, 0).unwrap();
        let spec = RegArimaSpec::new(crate::ArimaModel::new(order, order).unwrap());
        let mut fit = candidate(
            &[1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0],
            &[],
            TimeAxis::quarterly(2000, 1).unwrap(),
            spec.resolve(Transform::None).unwrap(),
            ArmaStart::Default,
            TimeAxis::quarterly(2000, 1).unwrap(),
            &[],
        )
        .unwrap();
        fit.innovation_variance = f64::INFINITY;
        assert_eq!(
            candidate_aicc(&fit, Transform::Log),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::TransformSelectionUnavailable {
                    candidate: Transform::Log,
                    reason: TransformSelectionUnavailableReason::NonFiniteScore,
                }
            )))
        );
        fit.innovation_variance = 1.0;
        fit.likelihood.as_mut().unwrap().aicc = None;
        assert_eq!(
            candidate_aicc(&fit, Transform::None),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::TransformSelectionUnavailable {
                    candidate: Transform::None,
                    reason: TransformSelectionUnavailableReason::InsufficientEffectiveObservations,
                }
            )))
        );
        fit.likelihood.as_mut().unwrap().aicc = Some(f64::NAN);
        assert_eq!(
            candidate_aicc(&fit, Transform::Log),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::TransformSelectionUnavailable {
                    candidate: Transform::Log,
                    reason: TransformSelectionUnavailableReason::NonFiniteScore,
                }
            )))
        );
    }
}
