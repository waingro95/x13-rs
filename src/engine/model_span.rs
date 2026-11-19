//! `arima.f` fits on modelspan, restores the end before forecasting, and
//! restores the beginning afterward. Restoration changes the design rows,
//! never the fitted coefficients, covariance, variance, or model residuals.

use crate::arima::{EstimatedRegArima, ExtensionsResult, generate_extensions};
use crate::error::NumericalDetail;
use crate::regarima::{ManualOutlier, OutlierType};
use crate::time::TimeAxis;
use crate::{Error, PeriodSpan, RegArimaSpec};

pub(super) struct EstimationInput<'a> {
    pub values: &'a [f64],
    pub missing: Vec<usize>,
    pub axis: TimeAxis,
    pub range: std::ops::Range<usize>,
    pub manual_outliers: Vec<ManualOutlier>,
    dropped_outliers: Vec<ManualOutlier>,
    has_model_span: bool,
}

impl<'a> EstimationInput<'a> {
    pub fn new(
        values: &'a [f64],
        missing: &[usize],
        axis: TimeAxis,
        request: &RegArimaSpec,
    ) -> Result<Self, Error> {
        let range = request.model_range(axis, values.len())?;
        if let Some(&index) = missing.iter().find(|&&index| !range.contains(&index)) {
            return Err(Error::MissingOutsideModelSpan {
                period: axis.period_at(index)?,
            });
        }
        if request.model_span().is_some() && !missing.is_empty() {
            // editor.f counts present seasons only within frstmd..endmd.
            let mut seasons = super::missing::SeasonCoverage::new(axis.frequency());
            for index in range.clone() {
                if !missing.contains(&index) {
                    seasons.observe(axis.period_at(index)?);
                }
            }
            seasons.require_all()?;
        }
        let model_axis = axis.shift(range.start as i64)?;
        let (manual_outliers, dropped_outliers) = if request.model_span().is_some() {
            ManualOutlier::validate_list(request.manual_outliers(), axis, values.len())?;
            let last = range.end.checked_sub(1).ok_or(Error::EmptyInput)?;
            let end = axis.period_at(last)?;
            // editor.f drops AO/TC outside modelspan and LS at or before its
            // beginning, warning once per regressor ("not within model span").
            // Both lists keep the supplied order.
            request
                .manual_outliers()
                .iter()
                .copied()
                .partition(|outlier| {
                    outlier.period() <= end
                        && if outlier.kind() == OutlierType::LevelShift {
                            outlier.period() > model_axis.start()
                        } else {
                            outlier.period() >= model_axis.start()
                        }
                })
        } else {
            (request.manual_outliers().to_vec(), Vec::new())
        };
        Ok(Self {
            values: &values[range.clone()],
            missing: missing
                .iter()
                .copied()
                .filter(|i| range.contains(i))
                .map(|i| i - range.start)
                .collect(),
            axis: model_axis,
            range,
            manual_outliers,
            dropped_outliers,
            has_model_span: request.model_span().is_some(),
        })
    }

    /// editor.f's "not within model span" warnings, after its backcast
    /// cancellation and in supplied order. editor.f deletes a manual AO or TC
    /// dated outside the span, and an LS dated after its end or not after its
    /// start.
    ///
    /// Only the explicit X-11 regARIMA and explicit SEATS pipelines call this:
    /// `RegArimaSpec::validate_model_policy` rejects manual outliers with an
    /// automatic model, transform or outlier search, so the automatic
    /// pipelines have none to drop.
    pub fn dropped_outlier_warnings(&self) -> impl Iterator<Item = crate::Warning> + '_ {
        self.dropped_outliers
            .iter()
            .map(|outlier| crate::Warning::ManualOutlierOutsideModelSpan {
                kind: outlier.kind(),
                period: outlier.period(),
            })
    }

    pub fn add_manual_outliers(
        &self,
        regression: &mut crate::regression::model::RegressionModel,
        spec: crate::api::ResolvedRegArimaSpec,
    ) -> Result<(), Error> {
        let rate = spec.resolved_temporary_change_rate(self.axis.frequency());
        if self.has_model_span {
            regression.add_validated_manual_outliers(
                &self.manual_outliers,
                self.axis,
                self.values.len(),
                rate,
            )
        } else {
            regression.add_manual_outliers(
                &self.manual_outliers,
                self.axis,
                self.values.len(),
                rate,
            )
        }
    }

    pub fn cancel_backcasts(
        &self,
        request: RegArimaSpec,
    ) -> Result<(RegArimaSpec, Option<crate::Warning>), Error> {
        let periods = request.forecast().backcast_periods();
        // editor.f cancels these before preparing the augmented X-11 span;
        // `RegArimaSpec::validate` applies the same rule.
        let extensions = request.forecast().for_model_start(self.range.start);
        if extensions == request.forecast() {
            return Ok((request, None));
        }
        Ok((
            request.with_forecast(extensions),
            Some(crate::Warning::ModelSpanBackcastsIgnored { periods }),
        ))
    }

    pub fn remove_easter_prior<'p>(
        &self,
        values: &mut [f64],
        full_prior: &'p [f64],
    ) -> Result<&'p [f64], Error> {
        let prior = &full_prior[self.range.clone()];
        remove_easter_prior(values, prior, self.axis)?;
        Ok(prior)
    }
}

fn remove_easter_prior(values: &mut [f64], prior: &[f64], axis: TimeAxis) -> Result<(), Error> {
    debug_assert_eq!(prior.len(), values.len());
    // The division and errors of `x11::easter::apply_prior`, dated on the
    // model span instead of the series start.
    for (index, (value, &factor)) in values.iter_mut().zip(prior).enumerate() {
        let period = Some(axis.period_at(index)?);
        if factor == 0.0 {
            return Err(Error::from(NumericalDetail::X11Easter {
                stage: crate::x11::X11EasterStage::Normalization,
                period,
                reason: crate::x11::X11EasterUnavailableReason::ZeroDivisor,
            }));
        }
        *value /= factor;
        if !value.is_finite() {
            return Err(Error::from(NumericalDetail::X11Easter {
                stage: crate::x11::X11EasterStage::Normalization,
                period,
                reason: crate::x11::X11EasterUnavailableReason::NonFiniteArithmetic,
            }));
        }
    }
    Ok(())
}

pub(super) fn apply_to_series(
    fit: &mut EstimatedRegArima,
    values: &[f64],
    axis: TimeAxis,
    easter_prior: Option<&[f64]>,
) -> Result<ExtensionsResult, Error> {
    debug_assert!(easter_prior.is_none_or(|prior| prior.len() == values.len()));
    if fit.axis == axis && fit.transformed.len() == values.len() {
        return generate_extensions(fit);
    }
    let last = values.len().checked_sub(1).ok_or(Error::EmptyInput)?;
    let start = PeriodSpan::new(fit.axis.start(), axis.period_at(last)?)
        .resolve_model_span(axis, values.len())?
        .start;
    // setspn/regvar before prtfct: preserve the fitted beginning and restore
    // all later observations. fcstxy reuses the fit's initial conditions.
    if start + fit.transformed.len() != values.len() {
        restore_response(
            fit,
            &values[start..],
            fit.axis,
            easter_prior.map(|prior| &prior[start..]),
        )?;
    }
    let extensions = generate_extensions(fit)?;
    if start != 0 {
        restore_response(fit, values, axis, easter_prior)?;
    }
    Ok(extensions)
}

fn restore_response(
    fit: &mut EstimatedRegArima,
    values: &[f64],
    axis: TimeAxis,
    easter_prior: Option<&[f64]>,
) -> Result<(), Error> {
    fit.regression_model = fit.regression_model.evaluated_over(axis, values.len())?;
    let mut adjusted = super::automatic_transform::remove_fixed_length(
        values,
        fit.regression_model.fixed_length(),
    );
    // x11pt1 removes fixed length before legacy Easter, including when the
    // length-of-period prior in trend mode overlaps the Easter months.
    if let Some(prior) = easter_prior {
        remove_easter_prior(&mut adjusted, prior, axis)?;
    }
    // x11pt1/setmv keep the placeholder after prior removal, and arima copies
    // that response again when it restores the model span, so the fitted
    // missing-value coefficient always meets the unscaled placeholder.
    for (kind, column) in fit.regression_model.columns() {
        if matches!(
            kind,
            crate::regression::model::RegressorKind::MissingValue { .. }
        ) {
            for (value, &indicator) in adjusted.iter_mut().zip(column) {
                if indicator != 0.0 {
                    *value = super::missing::PLACEHOLDER;
                }
            }
        }
    }
    fit.transformed = crate::transform::forward(&adjusted, fit.spec.transform())?;
    fit.axis = axis;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NumericalFailure;

    fn request() -> RegArimaSpec {
        let order = crate::ArimaOrder::new(0, 1, 1).unwrap();
        RegArimaSpec::new(crate::ArimaModel::new(order, order).unwrap())
    }

    /// The length-of-period prior is divided out before the Easter prior, and
    /// a gap keeps its placeholder.
    #[test]
    fn restored_response_divides_by_length_then_easter() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let order = crate::ArimaOrder::new(0, 0, 0).unwrap();
        let spec = RegArimaSpec::new(crate::ArimaModel::new(order, order).unwrap())
            .with_transform(crate::Transform::Log)
            .with_calendar(crate::regarima::CalendarSpec::new().with_basis(
                crate::regarima::CalendarBasis::FlowTradingDay(
                    crate::regarima::FlowTradingDay::new(
                        crate::regarima::TradingDayContrast::WeekdayWeekend,
                    ),
                ),
            ))
            .resolve_explicit()
            .unwrap()
            .with_flow_length_context(crate::api::FlowLengthContext::LengthOfPeriod);
        let mut values: Vec<_> = (0..72)
            .map(|i| 100.0 + i as f64 * 0.3 + (i as f64 * 0.7).sin())
            .collect();
        values[1] = super::super::missing::PLACEHOLDER;
        let mut regression =
            crate::regression::model::RegressionModel::from_spec(spec, axis, 48).unwrap();
        regression.add_missing(&[1], axis, 48).unwrap();
        let mut fitting = super::super::automatic_transform::remove_fixed_length(
            &values[..48],
            regression.fixed_length(),
        );
        super::super::missing::reset(&mut fitting, &[1]);
        let mut fit = crate::arima::estimate(
            &fitting,
            crate::arima::FitContext::unadjusted(&values[..48]),
            axis,
            spec,
            regression,
            crate::arima::ArmaStart::Default,
        )
        .unwrap();
        let easter_prior: Vec<_> = (0..values.len())
            .map(|i| if i % 12 == 2 { 1.073 } else { 1.0 })
            .collect();
        restore_response(&mut fit, &values, axis, Some(&easter_prior)).unwrap();
        assert_eq!(
            fit.transformed[1].to_bits(),
            super::super::missing::PLACEHOLDER.ln().to_bits()
        );
        for index in 0..values.len() {
            if index != 1 {
                let expected = (values[index]
                    / fit.regression_model.fixed_length()[index]
                    / easter_prior[index])
                    .ln();
                assert_eq!(fit.transformed[index].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn model_span_drops_manual_outliers_it_cannot_estimate() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let values = [100.0; 24];
        let outliers = [
            (OutlierType::Additive, 5),
            (OutlierType::Additive, 6),
            (OutlierType::TemporaryChange, 6),
            (OutlierType::LevelShift, 6),
            (OutlierType::LevelShift, 7),
            (OutlierType::LevelShift, 17),
            (OutlierType::TemporaryChange, 18),
            (OutlierType::LevelShift, 18),
        ]
        .map(|(kind, index)| ManualOutlier::new(kind, axis.period_at(index).unwrap()));
        let full_request = request().with_manual_outliers(&outliers);
        let full = EstimationInput::new(&values, &[], axis, &full_request).unwrap();
        assert_eq!(full.manual_outliers, outliers);
        let request = full_request.with_model_span(PeriodSpan::new(
            axis.period_at(6).unwrap(),
            axis.period_at(17).unwrap(),
        ));
        let input = EstimationInput::new(&values, &[], axis, &request).unwrap();
        let retained = [outliers[1], outliers[2], outliers[4], outliers[5]];
        assert_eq!(input.manual_outliers, retained);
        // editor.f warns once per dropped regressor, in supplied order.
        assert_eq!(full.dropped_outlier_warnings().count(), 0);
        let dropped = [outliers[0], outliers[3], outliers[6], outliers[7]];
        assert_eq!(
            input.dropped_outlier_warnings().collect::<Vec<_>>(),
            dropped.map(|outlier| crate::Warning::ManualOutlierOutsideModelSpan {
                kind: outlier.kind(),
                period: outlier.period(),
            })
        );
        let mut regression = crate::regression::model::RegressionModel::new();
        input
            .add_manual_outliers(&mut regression, request.resolve_explicit().unwrap())
            .unwrap();
        assert_eq!(regression.manual_outliers(), retained);
    }

    #[test]
    fn model_span_starting_late_cancels_the_backcasts() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let values = [100.0; 36];
        let original = request().with_forecast(
            crate::regarima::ForecastSpec::new(crate::regarima::ForecastHorizon::Periods(18), 12)
                .unwrap(),
        );
        for first in [0, 12] {
            let request = original.clone().with_model_span(PeriodSpan::new(
                axis.period_at(first).unwrap(),
                axis.period_at(35).unwrap(),
            ));
            let input = EstimationInput::new(&values, &[], axis, &request).unwrap();
            let (request, warning) = input.cancel_backcasts(request).unwrap();
            assert_eq!(request.forecast().forecast_periods(axis.frequency()), 18);
            assert_eq!(
                request.forecast().backcast_periods(),
                if first == 0 { 12 } else { 0 }
            );
            assert_eq!(
                warning,
                (first > 0).then_some(crate::Warning::ModelSpanBackcastsIgnored { periods: 12 })
            );
        }
    }

    #[test]
    fn gaps_are_checked_against_the_model_span() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let values = [100.0; 24];
        let request = request().with_model_span(PeriodSpan::new(
            axis.period_at(12).unwrap(),
            axis.period_at(23).unwrap(),
        ));
        assert!(matches!(
            EstimationInput::new(&values, &[0], axis, &request),
            Err(Error::MissingOutsideModelSpan { period }) if period == axis.start()
        ));
        assert!(matches!(
            EstimationInput::new(&values, &[12], axis, &request),
            Err(Error::MissingSeason {
                frequency: crate::Frequency::Monthly,
                subperiod: 1,
            })
        ));
    }

    #[test]
    fn easter_prior_division_reports_the_period_that_fails() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let values = [100.0; 24];
        for first in [0, 12] {
            let mut request = request();
            if first > 0 {
                request = request.with_model_span(PeriodSpan::new(
                    axis.period_at(first).unwrap(),
                    axis.period_at(23).unwrap(),
                ));
            }
            let input = EstimationInput::new(&values, &[], axis, &request).unwrap();
            let mut factors = [1.0; 24];
            factors[first] = 0.0;
            assert!(matches!(
                input.remove_easter_prior(&mut input.values.to_vec(), &factors),
                Err(Error::Numerical(NumericalFailure(NumericalDetail::X11Easter {
                    stage: crate::x11::X11EasterStage::Normalization,
                    period: Some(period),
                    reason: crate::x11::X11EasterUnavailableReason::ZeroDivisor,
                }))) if period == input.axis.start()
            ));
            factors[first] = 0.1;
            let mut fitting = input.values.to_vec();
            fitting[0] = f64::MAX;
            assert!(matches!(
                input.remove_easter_prior(&mut fitting, &factors),
                Err(Error::Numerical(NumericalFailure(NumericalDetail::X11Easter {
                    stage: crate::x11::X11EasterStage::Normalization,
                    period: Some(period),
                    reason: crate::x11::X11EasterUnavailableReason::NonFiniteArithmetic,
                }))) if period == input.axis.start()
            ));
        }
    }
}
