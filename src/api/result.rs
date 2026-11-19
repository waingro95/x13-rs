use super::Transform;
use super::regarima::{
    CalendarEffects, Convergence, DetectedOutlier, InformationCriteria, ParameterEstimate,
    ParameterKind, PrecisionCriterion, RegressionEstimate, RegressionVariable,
};
use crate::arima::{BackcastPoint, EstimatedRegArima, ForecastPoint};
use crate::regarima::{ArimaFactor, OutlierType, ResolvedArimaModel};
use crate::x11::CompletedComponents;
use crate::{Frequency, Period, Series};
use crate::{diagnostics, seats, x11};

/// The result of [`adjust`](crate::adjust): components, forecasts, the
/// fitted model, diagnostics and warnings.
///
/// The four components cover the observed span, one value per input
/// observation, and each is a [`Series`] starting at the input's first period
/// ([`start`](Adjustment::start)). The seasonally adjusted series and
/// trend-cycle are in series units; the seasonal and irregular components are
/// effects in series units for an additive decomposition and ratios near one
/// otherwise. What else is present depends on the pipeline:
///
/// | Accessor | Direct X-11 | regARIMA X-11 | SEATS |
/// |---|---|---|---|
/// | [`model`](Adjustment::model) | `None` | fitted model | fitted model |
/// | [`forecasts`](Adjustment::forecasts) | `None` | one year by default | one year by default |
/// | [`backcasts`](Adjustment::backcasts) | `None` | if requested | `None` |
/// | [`calendar_effects`](Adjustment::calendar_effects) | `None` | with calendar regressors | with calendar regressors |
/// | [`outliers`](Adjustment::outliers) | empty | fitted outliers | fitted outliers |
/// | [`diagnostics`](Adjustment::diagnostics) | X-11 | X-11 | SEATS |
///
/// Under X-11, a constant series skips model estimation, so it has no model
/// or forecasts ([`Warning::ConstantSeries`]); SEATS rejects it
/// ([`Error::DegenerateSeries`](crate::Error::DegenerateSeries)).
/// [`warnings`](Adjustment::warnings)
/// lists conditions X-13ARIMA-SEATS warns about but does not refuse.
///
/// ```
/// use x13::{Period, Series, Spec, adjust, x11};
///
/// let values: Vec<f64> = (0..48)
///     .map(|i| {
///         let season = 1.0 + 0.1 * (i as f64 * std::f64::consts::PI / 6.0).sin();
///         let noise = 1.0 + (((i * 37) % 11) as f64 / 11.0 - 0.5) / 50.0;
///         (100.0 + i as f64) * season * noise
///     })
///     .collect();
/// let series = Series::new(Period::monthly(2020, 1)?, &values);
/// let adjustment = adjust(series, &Spec::x11(x11::Options::new()))?;
///
/// let adjusted = adjustment.seasonally_adjusted();
/// let trend = adjustment.trend_cycle().expect("X-11 returns every component");
/// let irregular = adjustment.irregular();
/// assert_eq!(adjusted.len(), values.len());
/// assert_eq!(adjusted.start(), series.start());
/// // Multiplicative X-11: adjusted = trend × irregular.
/// let pairs = adjusted.values().iter().zip(trend.values()).zip(irregular.values());
/// for ((adjusted, trend), irregular) in pairs {
///     assert!((trend * irregular - adjusted).abs() < 1e-9 * adjusted);
/// }
/// for warning in adjustment.warnings() {
///     println!("warning: {warning}");
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Adjustment {
    pub(super) start: Period,
    pub(super) x11_easter: Option<x11::EasterEffect>,
    pub(super) seasonal: Vec<f64>,
    pub(super) seasonally_adjusted: Vec<f64>,
    pub(super) trend_cycle: Vec<f64>,
    pub(super) irregular: Vec<f64>,
    pub(super) seats_transitory: Option<seats::Transitory>,
    pub(super) model: Option<ModelResult>,
    pub(super) model_selection: Option<crate::regarima::ModelSelectionResult>,
    pub(super) bypass_transform_selection: Option<crate::regarima::TransformSelectionResult>,
    pub(super) forecasts: Option<Forecasts>,
    pub(super) backcasts: Option<Backcasts>,
    pub(super) calendar_effects: Option<CalendarEffects>,
    pub(super) outliers: Vec<DetectedOutlier>,
    pub(super) diagnostics: Diagnostics,
    pub(super) warnings: Vec<Warning>,
    pub(super) conditioning_scales: crate::x11::ConditioningScales,
}

impl Adjustment {
    pub(crate) fn set_seats_calendar_effects(&mut self, effects: CalendarEffects) {
        self.calendar_effects = Some(effects);
    }
    pub(crate) fn set_forecasts(&mut self, forecasts: Option<Forecasts>) {
        self.forecasts = forecasts;
    }
    /// The SEATS transitory component (table S14) over the observed span, when
    /// X-13ARIMA-SEATS produces one.
    /// Identity values have series units; log values are multiplicative ratios.
    #[must_use]
    pub fn seats_transitory(&self) -> Option<&seats::Transitory> {
        self.seats_transitory.as_ref()
    }

    pub(crate) fn set_seats_transitory(
        &mut self,
        component: seats::Transitory,
    ) -> Result<(), super::Error> {
        if component.series().len() != self.irregular.len() {
            return Err(super::Error::internal(
                "SEATS transitory component length differs from the series",
            ));
        }
        self.seats_transitory = Some(component);
        Ok(())
    }
    pub(crate) fn set_x11_easter(&mut self, result: x11::EasterEffect) {
        self.x11_easter = Some(result);
    }

    /// X-11's own Easter adjustment (the factors and how they were estimated),
    /// when requested with
    /// [`x11::Options::with_easter_estimation`](crate::x11::Options::with_easter_estimation);
    /// `None` otherwise.
    #[must_use]
    pub fn x11_easter(&self) -> Option<&x11::EasterEffect> {
        self.x11_easter.as_ref()
    }

    pub(crate) fn from_seats_components(
        start: Period,
        saved: seats::FinalComponents,
        model: ModelResult,
        diagnostics: seats::Diagnostics,
        warnings: Vec<Warning>,
    ) -> Self {
        Self {
            start,
            x11_easter: None,
            seasonal: saved.seasonal,
            seasonally_adjusted: saved.seasonally_adjusted,
            trend_cycle: saved.trend,
            irregular: saved.irregular,
            seats_transitory: None,
            model: Some(model),
            model_selection: None,
            bypass_transform_selection: None,
            forecasts: None,
            backcasts: None,
            calendar_effects: None,
            outliers: Vec::new(),
            diagnostics: Diagnostics {
                x11: None,
                seats: Some(diagnostics),
            },
            warnings,
            conditioning_scales: Default::default(),
        }
    }

    /// Adds what the regARIMA fit contributes to an X-11 result.
    pub(crate) fn set_regarima_outputs(
        &mut self,
        model: ModelResult,
        forecasts: Option<Forecasts>,
        backcasts: Option<Backcasts>,
        calendar_effects: Option<CalendarEffects>,
        outliers: Vec<DetectedOutlier>,
    ) {
        self.model = Some(model);
        self.forecasts = forecasts;
        self.backcasts = backcasts;
        self.calendar_effects = calendar_effects;
        self.outliers = outliers;
    }

    pub(crate) fn from_x11_components(
        start: Period,
        completed: CompletedComponents,
        mut warnings: Vec<Warning>,
    ) -> Self {
        let CompletedComponents {
            components,
            quality,
            spectral,
        } = completed;
        warnings.extend(components.warnings);
        Self {
            start,
            seasonal: components.seasonal,
            x11_easter: None,
            seasonally_adjusted: components.seasonally_adjusted,
            trend_cycle: components.trend_cycle,
            irregular: components.irregular,
            seats_transitory: None,
            model: None,
            model_selection: None,
            bypass_transform_selection: None,
            forecasts: None,
            backcasts: None,
            calendar_effects: None,
            outliers: Vec::new(),
            diagnostics: Diagnostics {
                seats: None,
                x11: Some(x11::Diagnostics {
                    decomposition: components.decomposition,
                    mode: components.mode,
                    final_henderson_terms: components.final_henderson_terms,
                    msr_selection: components.msr_selection,
                    calendar_sigma_selection: components.calendar_sigma_selection,
                    seasonality: components.seasonality,
                    quality,
                    spectral,
                }),
            },
            warnings,
            conditioning_scales: components.conditioning_scales,
        }
    }

    pub(crate) fn set_model_selection(&mut self, outcome: crate::regarima::ModelSelectionResult) {
        self.model_selection = Some(outcome);
    }

    pub(crate) fn set_outliers(&mut self, outliers: Vec<DetectedOutlier>) {
        self.outliers = outliers;
    }

    pub(crate) fn set_bypass_transform_selection(
        &mut self,
        selection: Option<crate::regarima::TransformSelectionResult>,
    ) {
        self.bypass_transform_selection = selection;
    }

    /// The first period of the input series, where every component starts.
    #[must_use]
    pub const fn start(&self) -> Period {
        self.start
    }

    /// Final seasonal component over the observed span (table D10; SEATS
    /// table S10), dated from [`start`](Adjustment::start).
    ///
    /// Additive X-11 and SEATS without a transform give effects in series
    /// units. Multiplicative, log-additive and pseudo-additive X-11 and log
    /// SEATS give factors: ratios near one, not percentages or logarithms.
    /// Trend-only X-11 gives neutral values, zero if additive and one
    /// otherwise. Pseudo-additive X-11 adjusts as
    /// `original - trend × (factor - 1)`.
    ///
    /// X-11 always returns `Some`. SEATS returns `None` where
    /// X-13ARIMA-SEATS omits table S10: the model has no seasonal part and the
    /// series is not transformed, so the component is zero throughout.
    #[must_use]
    pub fn seasonal(&self) -> Option<Series<'_>> {
        // SEATS leaves S10 empty exactly where Build 62 omits it
        // (USRENTRY IFUNC=1201); X-11 components always span the input.
        (!self.seasonal.is_empty()).then(|| Series::new(self.start, &self.seasonal))
    }

    /// Final seasonally adjusted series over the observed span (table D11;
    /// SEATS table S11), in series units, dated from
    /// [`start`](Adjustment::start).
    ///
    /// Trend-only X-11 removes no seasonal variation, but calendar, prior and
    /// final-outlier adjustments still apply.
    ///
    /// The slice from [`Series::values`] borrows the adjustment, not the
    /// view, so it outlives the temporary `Series`:
    ///
    /// ```
    /// use x13::{Period, Series, Spec, adjust, x11};
    ///
    /// let values: Vec<f64> = (0..48)
    ///     .map(|i| (100.0 + i as f64) * (1.0 + 0.1 * (i as f64 * std::f64::consts::PI / 6.0).sin()))
    ///     .collect();
    /// let start = Period::monthly(2020, 1)?;
    /// let spec = Spec::x11(x11::Options::new());
    /// let adjustment = adjust(Series::new(start, &values), &spec)?;
    ///
    /// let adjusted: &[f64] = adjustment.seasonally_adjusted().values();
    /// assert_eq!(adjusted.len(), values.len());
    /// assert_eq!(adjustment.seasonally_adjusted().start(), start);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn seasonally_adjusted(&self) -> Series<'_> {
        Series::new(self.start, &self.seasonally_adjusted)
    }

    /// Final trend-cycle over the observed span (table D12; SEATS table S12),
    /// in series units, dated from [`start`](Adjustment::start).
    ///
    /// The SEATS trend-cycle excludes the transitory component
    /// ([`seats_transitory`](Adjustment::seats_transitory)), so SEATS
    /// components need not add or multiply back exactly to the series.
    ///
    /// X-11 always returns `Some`. SEATS returns `None` where
    /// X-13ARIMA-SEATS omits table S12: the final trend, restored trend
    /// effects included, is zero throughout.
    #[must_use]
    pub fn trend_cycle(&self) -> Option<Series<'_>> {
        // SEATS leaves S12 empty exactly where Build 62 omits it
        // (ansub9 USRENTRY1310); X-11 components always span the input.
        (!self.trend_cycle.is_empty()).then(|| Series::new(self.start, &self.trend_cycle))
    }

    /// Final irregular component over the observed span (table D13; SEATS
    /// table S13): effects in series units for additive X-11 and SEATS
    /// without a transform, ratios near one otherwise (not percentages or
    /// logarithms). Dated from [`start`](Adjustment::start).
    #[must_use]
    pub fn irregular(&self) -> Series<'_> {
        Series::new(self.start, &self.irregular)
    }

    /// The fitted regARIMA model, or `None` for direct X-11 and for a
    /// constant series under X-11, where estimation is skipped
    /// ([`Warning::ConstantSeries`]).
    #[must_use]
    pub fn model(&self) -> Option<&ModelResult> {
        self.model.as_ref()
    }

    /// Outcome of the automatic model search
    /// ([`RegArimaSpec::automatic_model`](crate::RegArimaSpec::automatic_model)),
    /// or `None` for a fixed model or direct X-11.
    #[must_use]
    pub fn model_selection(&self) -> Option<&crate::regarima::ModelSelectionResult> {
        self.model_selection.as_ref()
    }

    /// Outcome of the automatic transform choice, or `None` for a fixed
    /// transform. Present also when a zero or negative value decided it
    /// without fitting.
    pub fn transform_selection(&self) -> Option<&crate::regarima::TransformSelectionResult> {
        self.model
            .as_ref()
            .and_then(ModelResult::transform_selection)
            .or(self.bypass_transform_selection.as_ref())
    }

    /// Forecasts from the fitted regARIMA model, starting the period after
    /// the last observation.
    ///
    /// `None` for direct X-11, a zero forecast horizon, or a series with no
    /// fitted model. Forecasts are returned even when X-11 cannot use them
    /// ([`Warning::ForecastsNotUsed`]). Under SEATS they come from the
    /// regARIMA [`ModelResult`], not from any model SEATS re-estimates.
    #[must_use]
    pub fn forecasts(&self) -> Option<&Forecasts> {
        self.forecasts.as_ref()
    }

    /// Backcasts from the fitted regARIMA model, in calendar order and ending
    /// the period before the first observation.
    ///
    /// `None` unless backcasts were requested under regARIMA X-11: SEATS
    /// ignores them, and a model span that starts after the first
    /// observation cancels them.
    #[must_use]
    pub fn backcasts(&self) -> Option<&Backcasts> {
        self.backcasts.as_ref()
    }

    /// Calendar effects over the observed span, when the model has calendar
    /// regressors (built-in, or user trading-day and holiday columns).
    ///
    /// Under SEATS they come from the regARIMA fit and stay separate from
    /// the seasonal component; the combined adjustment factors (table S16)
    /// are not returned, so the series is not in general the seasonal
    /// component times (or plus) the adjusted series.
    #[must_use]
    pub fn calendar_effects(&self) -> Option<&CalendarEffects> {
        self.calendar_effects.as_ref()
    }

    /// Outliers in the final regARIMA model, specified or found by the search,
    /// in the order X-13ARIMA-SEATS reports them. Empty for direct X-11.
    #[must_use]
    pub fn outliers(&self) -> &[DetectedOutlier] {
        &self.outliers
    }

    /// Diagnostics for this adjustment: [`Diagnostics::x11`] for the X-11
    /// pipelines, [`Diagnostics::seats`] for SEATS.
    #[must_use]
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// Conditions X-13ARIMA-SEATS warns about but does not refuse, such as a
    /// seasonal filter replaced because the series is short, or forecasts
    /// X-11 could not use. The adjustment is still complete.
    #[must_use]
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// X-11 D12 and D13 Henderson operation scales over the observed span,
    /// in that order. A development hook, not part of the public API. Empty
    /// for SEATS, log-additive and pseudo-additive runs.
    #[cfg(feature = "__conditioning-scales")]
    #[doc(hidden)]
    #[must_use]
    pub fn conditioning_scales(&self) -> (&[f64], &[f64]) {
        (
            &self.conditioning_scales.trend_cycle,
            &self.conditioning_scales.irregular,
        )
    }
}

// Checks that no `Ok` output is NaN or infinite. Build 62 prints and continues
// past +Infinity (PRTFCT's INVFCN on an overflowing bound) and HQ = -Infinity
// (PRLKHD's log(log(1)) at one effective observation). Read-only.
impl Adjustment {
    pub(super) fn validate_finite(&self) -> Result<(), crate::Error> {
        use super::{AdjustmentOutput as O, Component as C};
        for (output, values) in [
            (O::Component(C::Seasonal), &self.seasonal),
            (
                O::Component(C::SeasonallyAdjusted),
                &self.seasonally_adjusted,
            ),
            (O::Component(C::TrendCycle), &self.trend_cycle),
            (O::Component(C::Irregular), &self.irregular),
        ] {
            finite(output, values.iter().copied())?;
        }
        if let Some(transitory) = &self.seats_transitory {
            finite(O::Transitory, transitory.values.iter().copied())?;
        }
        if let Some(forecasts) = &self.forecasts {
            for (output, values) in [
                (O::ForecastValue, &forecasts.values),
                (O::ForecastLower95, &forecasts.lower_95),
                (O::ForecastUpper95, &forecasts.upper_95),
                (O::ForecastTransformedValue, &forecasts.transformed_values),
                (
                    O::ForecastTransformedStandardError,
                    &forecasts.transformed_standard_errors,
                ),
                (
                    O::ForecastRegressionVariance,
                    &forecasts.regression_variances,
                ),
                (
                    O::ForecastStochasticVariance,
                    &forecasts.stochastic_variances,
                ),
            ] {
                finite(output, values.iter().copied())?;
            }
        }
        if let Some(backcasts) = &self.backcasts {
            for (output, values) in [
                (O::BackcastValue, &backcasts.values),
                (O::BackcastLower95, &backcasts.lower_95),
                (O::BackcastUpper95, &backcasts.upper_95),
                (O::BackcastTransformedValue, &backcasts.transformed_values),
                (
                    O::BackcastTransformedStandardError,
                    &backcasts.transformed_standard_errors,
                ),
            ] {
                finite(output, values.iter().copied())?;
            }
        }
        if let Some(model) = &self.model {
            model.validate_finite()?;
        }
        // An absent standard error or t-value is published as `None`; zero
        // keeps the list position as the reported index, here and for models.
        finite(O::OutlierEstimate, self.outliers.iter().map(|o| o.estimate))?;
        finite(
            O::OutlierStandardError,
            self.outliers
                .iter()
                .map(|o| o.standard_error.unwrap_or(0.0)),
        )?;
        finite(
            O::OutlierTValue,
            self.outliers.iter().map(|o| o.t_value.unwrap_or(0.0)),
        )?;
        if let Some(calendar) = &self.calendar_effects {
            for (output, values) in [
                (O::CalendarEstimated, &calendar.estimated),
                (O::CalendarFixedLength, &calendar.fixed_length),
                (O::CalendarCombined, &calendar.combined),
            ] {
                finite(output, values.iter().copied())?;
            }
        }
        Ok(())
    }
}

fn finite(
    output: super::AdjustmentOutput,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), crate::Error> {
    match values.into_iter().position(|value| !value.is_finite()) {
        Some(index) => Err(crate::Error::from(
            crate::error::NumericalDetail::NonFiniteOutput { output, index },
        )),
        None => Ok(()),
    }
}

impl ModelResult {
    fn validate_finite(&self) -> Result<(), crate::Error> {
        use super::AdjustmentOutput as O;
        finite(
            O::ParameterEstimate,
            self.parameters.iter().map(|p| p.estimate),
        )?;
        finite(
            O::ParameterStandardError,
            self.parameters
                .iter()
                .map(|p| p.standard_error.unwrap_or(0.0)),
        )?;
        finite(
            O::RegressionEstimate,
            self.regression_coefficients.iter().map(|r| r.estimate),
        )?;
        finite(
            O::RegressionStandardError,
            self.regression_coefficients
                .iter()
                .map(|r| r.standard_error.unwrap_or(0.0)),
        )?;
        finite(O::InnovationVariance, [self.innovation_variance])?;
        finite(
            O::LogDeterminantCorrection,
            [self.log_determinant_correction],
        )?;
        finite(O::LogLikelihood, self.log_likelihood)?;
        if let Some(criteria) = &self.information_criteria {
            finite(O::Aic, [criteria.aic])?;
            finite(O::Aicc, criteria.aicc)?;
            finite(O::HannanQuinn, [criteria.hannan_quinn])?;
            finite(O::Bic, [criteria.bic])?;
        }
        if let ArmaCovariance::Calculated(values) = &self.arma_covariance {
            finite(O::ArmaCovariance, values.iter().copied())?;
        }
        if let Some(values) = &self.regression_covariance {
            finite(O::RegressionCovariance, values.iter().copied())?;
        }
        finite(O::Residual, self.residuals.iter().copied())
    }
}

/// The fitted regARIMA model, from [`Adjustment::model`].
///
/// It holds the model's orders and transform, the ARMA and regression
/// estimates with standard errors, the likelihood and information criteria,
/// residuals and iteration counts. With automatic selection it describes
/// the chosen model.
///
/// ```
/// use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, adjust, x11};
///
/// let spec = Spec::x11_regarima(
///     RegArimaSpec::new(ArimaModel::airline()).with_transform(Transform::Log),
///     x11::Options::new(),
/// );
/// let values: Vec<f64> = (0..72)
///     .map(|i| {
///         let season = 1.0 + 0.1 * (i as f64 * std::f64::consts::PI / 6.0).sin();
///         let noise = 1.0 + (((i * 37) % 11) as f64 / 11.0 - 0.5) / 50.0;
///         (100.0 + i as f64) * season * noise
///     })
///     .collect();
/// let series = Series::new(Period::monthly(2015, 1)?, &values);
/// let adjustment = adjust(series, &spec)?;
/// let model = adjustment.model().expect("a fitted model");
///
/// // The airline model has one regular and one seasonal MA coefficient.
/// assert_eq!(model.parameters().len(), 2);
/// assert!(model.parameters().iter().all(|p| p.estimate().abs() < 1.0));
/// assert!(model.information_criteria().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResult {
    transform: Transform,
    transform_selection: Option<crate::regarima::TransformSelectionResult>,
    model: ResolvedArimaModel,
    parameters: Vec<ParameterEstimate>,
    regression_coefficients: Vec<RegressionEstimate>,
    innovation_variance: f64,
    log_determinant_correction: f64,
    log_likelihood: Option<f64>,
    information_criteria: Option<InformationCriteria>,
    arma_covariance: ArmaCovariance,
    regression_covariance: Option<Vec<f64>>,
    residuals: Vec<f64>,
    residual_spectral: Result<diagnostics::SpectralPeaks, diagnostics::Unavailable>,
    convergence: Convergence,
}

/// Outcome of the fitted model's ARMA covariance calculation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArmaCovariance {
    /// No ARMA parameters require covariance estimation.
    NotRequired,
    /// Successfully calculated covariance, including valid all-zero matrices.
    Calculated(Vec<f64>),
    /// Singular information matrix; fitted coefficients remain usable.
    Singular,
}

impl ArmaCovariance {
    /// The calculated matrix, row-major, when there is one.
    pub(crate) fn matrix(&self) -> Option<&[f64]> {
        match self {
            Self::Calculated(matrix) => Some(matrix),
            Self::NotRequired | Self::Singular => None,
        }
    }
}

// The estimator retains the source packed triangle. Public covariance uses
// the documented symmetric row-major layout without changing any arithmetic.
fn regression_covariance_row_major(packed: &[f64], count: usize) -> Vec<f64> {
    debug_assert_eq!(packed.len(), count * (count + 1) / 2);
    (0..count)
        .flat_map(|row| {
            (0..count).map(move |column| {
                let low = row.min(column);
                let high = row.max(column);
                packed[high * (high + 1) / 2 + low]
            })
        })
        .collect()
}

impl ModelResult {
    /// The public result of `estimated`. `residual_override` replaces the
    /// fit's residuals where Build 62 reports another fit's: automdl keeps the
    /// original model's residuals when a fallback refits a saved model.
    pub(crate) fn from_estimated(
        estimated: EstimatedRegArima,
        residual_override: Option<Vec<f64>>,
        residual_spectral: Result<diagnostics::SpectralPeaks, diagnostics::Unavailable>,
    ) -> Self {
        let spec = estimated.spec;
        let frequency = estimated.axis.frequency();
        let parameters = parameter_layout(spec.model(), frequency)
            .into_iter()
            .zip(estimated.parameters.into_iter().enumerate())
            .map(
                |((kind, factor, period, lag), (index, estimate))| ParameterEstimate {
                    kind,
                    factor,
                    period,
                    lag,
                    estimate,
                    standard_error: estimated
                        .parameter_standard_errors
                        .as_ref()
                        .map(|errors| errors[index]),
                },
            )
            .collect();
        let regression_count = estimated.regression_coefficients.len();
        let regression_covariance = estimated
            .regression_covariance
            .as_ref()
            .map(|packed| regression_covariance_row_major(packed, regression_count));
        let regression_coefficients = estimated
            .regressor_kinds
            .into_iter()
            .zip(estimated.regression_coefficients.into_iter().enumerate())
            .map(|(kind, (index, estimate))| RegressionEstimate {
                variable: kind.variable(),
                name: kind.name(),
                estimate,
                standard_error: estimated
                    .regression_standard_errors
                    .as_ref()
                    .map(|errors| errors[index]),
            })
            .collect();
        let likelihood = estimated.likelihood;
        Self {
            transform: spec.transform(),
            transform_selection: None,
            model: spec.model(),
            parameters,
            regression_coefficients,
            innovation_variance: estimated.innovation_variance,
            log_determinant_correction: estimated.log_determinant_correction,
            log_likelihood: likelihood.map(|value| value.log_likelihood),
            information_criteria: likelihood.map(|likelihood| InformationCriteria {
                aic: likelihood.aic,
                aicc: likelihood.aicc,
                hannan_quinn: likelihood.hannan_quinn,
                bic: likelihood.bic,
            }),
            arma_covariance: estimated.arma_covariance,
            regression_covariance,
            residuals: residual_override.unwrap_or(estimated.residuals),
            residual_spectral,
            convergence: Convergence {
                overall_iterations: estimated.overall_iterations,
                nonlinear_iterations: estimated.nonlinear_iterations,
            },
        }
    }

    /// Transformation used during estimation.
    #[must_use]
    pub const fn transform(&self) -> Transform {
        self.transform
    }

    /// Outcome of the automatic transform choice, or `None` for a fixed
    /// transform.
    #[must_use]
    pub const fn transform_selection(&self) -> Option<&crate::regarima::TransformSelectionResult> {
        self.transform_selection.as_ref()
    }

    pub(crate) fn set_transform_selection(
        &mut self,
        selection: Option<crate::regarima::TransformSelectionResult>,
    ) {
        self.transform_selection = selection;
    }

    /// Orders of the fitted ARIMA model; displays as `(p d q)(P D Q)`.
    #[must_use]
    pub const fn model(&self) -> ResolvedArimaModel {
        self.model
    }

    /// Estimated AR and MA coefficients, in the order X-13ARIMA-SEATS reports
    /// them.
    #[must_use]
    pub fn parameters(&self) -> &[ParameterEstimate] {
        &self.parameters
    }

    /// Estimated regression coefficients, in the order X-13ARIMA-SEATS reports
    /// them.
    #[must_use]
    pub fn regression_coefficients(&self) -> &[RegressionEstimate] {
        &self.regression_coefficients
    }

    /// Whether the final regression design contains a constant term.
    #[must_use]
    pub fn includes_constant(&self) -> bool {
        self.regression_coefficients
            .iter()
            .any(|coefficient| coefficient.variable() == RegressionVariable::Constant)
    }

    /// Final innovation variance.
    #[must_use]
    pub const fn innovation_variance(&self) -> f64 {
        self.innovation_variance
    }

    /// The likelihood's log-determinant term as X-13ARIMA-SEATS reports it,
    /// after the covariance step, also when the innovation variance is zero.
    ///
    /// [`log_likelihood`](Self::log_likelihood) uses the value from
    /// estimation, so it need not be reproducible from this one.
    #[must_use]
    pub const fn log_determinant_correction(&self) -> f64 {
        self.log_determinant_correction
    }

    /// Final maximized log likelihood, unavailable for zero innovation variance.
    #[must_use]
    pub const fn log_likelihood(&self) -> Option<f64> {
        self.log_likelihood
    }

    /// AIC-family criteria, unavailable when the likelihood is unavailable.
    #[must_use]
    pub const fn information_criteria(&self) -> Option<&InformationCriteria> {
        self.information_criteria.as_ref()
    }

    /// Row-major covariance of the ARMA parameter estimates. `None` when the
    /// model has no ARMA parameters, or when the matrix is singular, which
    /// also raises [`Warning::SingularArmaCovariance`].
    #[must_use]
    pub fn arma_covariance(&self) -> Option<&[f64]> {
        self.arma_covariance.matrix()
    }

    /// Row-major regression covariance, when regression uncertainty is available.
    #[must_use]
    pub fn regression_covariance(&self) -> Option<&[f64]> {
        self.regression_covariance.as_deref()
    }

    /// Residuals of the final fit, as X-13ARIMA-SEATS reports them.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Spectral peaks in the final fit's residuals, from the default
    /// autoregressive spectrum, whichever adjustment method follows. Monthly
    /// series only.
    pub const fn residual_spectral(
        &self,
    ) -> Result<&diagnostics::SpectralPeaks, &diagnostics::Unavailable> {
        self.residual_spectral.as_ref()
    }

    /// Iteration counts and convergence status.
    #[must_use]
    pub const fn convergence(&self) -> &Convergence {
        &self.convergence
    }
}

/// One forecast or backcast point's columns that [`Forecasts`] and
/// [`Backcasts`] share.
struct ExtensionPoint {
    period: Period,
    original: f64,
    lower_95: f64,
    upper_95: f64,
    transformed: f64,
    transformed_standard_error: f64,
}

/// The columns [`Forecasts`] and [`Backcasts`] share, in point order.
struct ExtensionColumns {
    start: Period,
    values: Vec<f64>,
    lower_95: Vec<f64>,
    upper_95: Vec<f64>,
    transformed_values: Vec<f64>,
    transformed_standard_errors: Vec<f64>,
}

impl ExtensionColumns {
    /// The columns of `points`, starting at the first one's period; `None`
    /// without points.
    fn collect(points: impl ExactSizeIterator<Item = ExtensionPoint>) -> Option<Self> {
        let mut points = points.peekable();
        let start = points.peek()?.period;
        let count = points.len();
        let mut columns = Self {
            start,
            values: Vec::with_capacity(count),
            lower_95: Vec::with_capacity(count),
            upper_95: Vec::with_capacity(count),
            transformed_values: Vec::with_capacity(count),
            transformed_standard_errors: Vec::with_capacity(count),
        };
        for point in points {
            columns.values.push(point.original);
            columns.lower_95.push(point.lower_95);
            columns.upper_95.push(point.upper_95);
            columns.transformed_values.push(point.transformed);
            columns
                .transformed_standard_errors
                .push(point.transformed_standard_error);
        }
        Some(columns)
    }
}

/// Forecasts from the fitted regARIMA model, with 95 percent limits, from
/// [`Adjustment::forecasts`].
///
/// Forecast `i` falls `i` periods after [`start`](Forecasts::start), the
/// period after the last observation. Point forecasts and limits are on the
/// original scale; the transformed-scale values and standard errors are
/// the model's own, before the transform and any prior adjustments (such as
/// leap-year or X-11 Easter factors) are undone.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecasts {
    start: Period,
    values: Vec<f64>,
    lower_95: Vec<f64>,
    upper_95: Vec<f64>,
    transformed_values: Vec<f64>,
    transformed_standard_errors: Vec<f64>,
    regression_variances: Vec<f64>,
    stochastic_variances: Vec<f64>,
}

impl Forecasts {
    pub(crate) fn from_points(points: Vec<ForecastPoint>) -> Option<Self> {
        let ExtensionColumns {
            start,
            values,
            lower_95,
            upper_95,
            transformed_values,
            transformed_standard_errors,
        } = ExtensionColumns::collect(points.iter().map(|point| ExtensionPoint {
            period: point.period,
            original: point.original,
            lower_95: point.lower_95,
            upper_95: point.upper_95,
            transformed: point.transformed,
            transformed_standard_error: point.transformed_standard_error,
        }))?;
        // The variance split exists for the points that carry one.
        let (regression_variances, stochastic_variances) = points
            .iter()
            .filter_map(|point| {
                point
                    .regression_variance
                    .map(|regression| (regression, point.stochastic_variance))
            })
            .unzip();
        Some(Self {
            start,
            values,
            lower_95,
            upper_95,
            transformed_values,
            transformed_standard_errors,
            regression_variances,
            stochastic_variances,
        })
    }

    /// First forecast period.
    #[must_use]
    pub const fn start(&self) -> Period {
        self.start
    }

    /// Original-scale point forecasts, dated from [`start`](Self::start).
    #[must_use]
    pub fn series(&self) -> Series<'_> {
        Series::new(self.start, &self.values)
    }

    /// Original-scale lower 95-percent limits, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn lower_95(&self) -> Series<'_> {
        Series::new(self.start, &self.lower_95)
    }

    /// Original-scale upper 95-percent limits, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn upper_95(&self) -> Series<'_> {
        Series::new(self.start, &self.upper_95)
    }

    /// Transformed-scale point forecasts, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn transformed(&self) -> Series<'_> {
        Series::new(self.start, &self.transformed_values)
    }

    /// Transformed-scale forecast standard errors.
    #[must_use]
    pub fn transformed_standard_errors(&self) -> &[f64] {
        &self.transformed_standard_errors
    }

    /// Regression contribution to forecast variance, when applicable.
    #[must_use]
    pub fn regression_variances(&self) -> &[f64] {
        &self.regression_variances
    }

    /// Stochastic contribution to forecast variance, when applicable.
    #[must_use]
    pub fn stochastic_variances(&self) -> &[f64] {
        &self.stochastic_variances
    }
}

/// Backcasts from the fitted regARIMA model, with 95 percent limits, from
/// [`Adjustment::backcasts`].
///
/// Values are in calendar order: the first falls at
/// [`start`](Backcasts::start) and the last in the period before the first
/// observation. Scales are as for [`Forecasts`].
#[derive(Debug, Clone, PartialEq)]
pub struct Backcasts {
    start: Period,
    values: Vec<f64>,
    lower_95: Vec<f64>,
    upper_95: Vec<f64>,
    transformed_values: Vec<f64>,
    transformed_standard_errors: Vec<f64>,
}

impl Backcasts {
    pub(crate) fn from_points(points: Vec<BackcastPoint>) -> Option<Self> {
        let ExtensionColumns {
            start,
            values,
            lower_95,
            upper_95,
            transformed_values,
            transformed_standard_errors,
        } = ExtensionColumns::collect(points.into_iter().map(|point| ExtensionPoint {
            period: point.period,
            original: point.original,
            lower_95: point.lower_95,
            upper_95: point.upper_95,
            transformed: point.transformed,
            transformed_standard_error: point.transformed_standard_error,
        }))?;
        Some(Self {
            start,
            values,
            lower_95,
            upper_95,
            transformed_values,
            transformed_standard_errors,
        })
    }

    /// First chronological backcast period.
    #[must_use]
    pub const fn start(&self) -> Period {
        self.start
    }

    /// Original-scale point backcasts, dated from [`start`](Self::start).
    #[must_use]
    pub fn series(&self) -> Series<'_> {
        Series::new(self.start, &self.values)
    }

    /// Original-scale lower 95-percent limits, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn lower_95(&self) -> Series<'_> {
        Series::new(self.start, &self.lower_95)
    }

    /// Original-scale upper 95-percent limits, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn upper_95(&self) -> Series<'_> {
        Series::new(self.start, &self.upper_95)
    }

    /// Transformed-scale point backcasts, dated like
    /// [`series`](Self::series).
    #[must_use]
    pub fn transformed(&self) -> Series<'_> {
        Series::new(self.start, &self.transformed_values)
    }

    /// Transformed-scale backcast standard errors.
    #[must_use]
    pub fn transformed_standard_errors(&self) -> &[f64] {
        &self.transformed_standard_errors
    }
}

fn parameter_layout(
    model: ResolvedArimaModel,
    frequency: Frequency,
) -> Vec<(ParameterKind, ArimaFactor, u8, u8)> {
    let seasonal_period = frequency.periods_per_year();
    let mut layout = Vec::new();
    for (kind, factor, order, period) in [
        (
            ParameterKind::Ar,
            ArimaFactor::Regular,
            model.regular().p(),
            1,
        ),
        (
            ParameterKind::Ar,
            ArimaFactor::Seasonal,
            model.seasonal().p(),
            seasonal_period,
        ),
        (
            ParameterKind::Ma,
            ArimaFactor::Regular,
            model.regular().q(),
            1,
        ),
        (
            ParameterKind::Ma,
            ArimaFactor::Seasonal,
            model.seasonal().q(),
            seasonal_period,
        ),
    ] {
        for factor_lag in 1..=order {
            layout.push((kind, factor, period, factor_lag * period));
        }
    }
    layout
}

/// Diagnostics of an adjustment, from [`Adjustment::diagnostics`]: X-11
/// diagnostics for the X-11 pipelines, SEATS diagnostics for SEATS.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub(super) x11: Option<x11::Diagnostics>,
    pub(super) seats: Option<seats::Diagnostics>,
}

impl Diagnostics {
    /// Diagnostics for a SEATS adjustment, when SEATS was used.
    #[must_use]
    pub fn seats(&self) -> Option<&seats::Diagnostics> {
        self.seats.as_ref()
    }
    /// Diagnostics for an X-11 adjustment, when X-11 was used.
    #[must_use]
    pub fn x11(&self) -> Option<&x11::Diagnostics> {
        self.x11.as_ref()
    }
}

/// A condition X-13ARIMA-SEATS warns about but does not refuse, from
/// [`Adjustment::warnings`].
///
/// The adjustment is complete, but some warnings change how to read it:
/// X-11 could not use the forecasts
/// ([`ForecastsNotUsed`](Warning::ForecastsNotUsed)), a seasonal filter
/// was replaced for a short series
/// ([`SeasonalFilterReplaced`](Warning::SeasonalFilterReplaced)), or a
/// spectrum shows seasonal or trading-day peaks left in the result
/// ([`ResidualSpectralPeaks`](Warning::ResidualSpectralPeaks),
/// [`X11SpectralPeaks`](Warning::X11SpectralPeaks)). `Display` gives a
/// one-line message. More variants may be added.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Warning {
    /// X-13ARIMA-SEATS does not generate requested backcasts for SEATS
    /// adjustments.
    #[non_exhaustive]
    SeatsBackcastsIgnored {
        /// Number of backcasts requested.
        periods: u16,
    },
    /// X-13ARIMA-SEATS cancels backcasts when the model span starts after the
    /// first observation.
    #[non_exhaustive]
    ModelSpanBackcastsIgnored {
        /// Number of backcasts requested.
        periods: u16,
    },
    /// A manual outlier the model span cannot fit was dropped:
    /// an additive outlier or temporary change dated outside the span, or a
    /// level shift dated at or before its first period or after its last.
    /// One warning per dropped outlier, in supplied order.
    #[non_exhaustive]
    ManualOutlierOutsideModelSpan {
        /// Type of the dropped outlier.
        kind: OutlierType,
        /// Period of the dropped outlier.
        period: Period,
    },
    /// The spectrum of the final regARIMA residuals has seasonal or
    /// trading-day peaks.
    #[non_exhaustive]
    ResidualSpectralPeaks {
        /// Kind of peaks found.
        kind: diagnostics::SpectralPeakKind,
    },
    /// X-11 spectral diagnostics found seasonal or trading-day peaks.
    #[non_exhaustive]
    X11SpectralPeaks {
        /// Kind of peaks found.
        kind: diagnostics::SpectralPeakKind,
    },
    /// The spectrum of the original series has no seasonal peak (checked
    /// only for a seasonal, not trend-only, X-11 decomposition).
    X11NoOriginalSeasonalSpectralPeaks,
    /// A seasonality test (tables B, D, final D11 or E2) has no value
    /// because the original series is constant or the residual mean square
    /// is zero. The test's accessor on
    /// [`diagnostics::SeasonalityTests`](super::diagnostics::SeasonalityTests)
    /// returns the reason.
    #[non_exhaustive]
    SeasonalityTestUnavailable {
        /// Test that has no value.
        test: super::diagnostics::SeasonalityTest,
    },
    /// No regARIMA forecasts were requested for pseudo-additive seasonal
    /// difference forecasting; observed components remain available.
    PseudoAdditiveWithoutForecasts,
    /// Preprocessing was skipped because the original observations are all
    /// exactly equal.
    ConstantSeries,
    /// The likelihood optimizer reported improper input.
    ImproperLikelihoodInput,
    /// The optimizer could not improve the likelihood further, by the
    /// reported criterion; the fit is kept.
    #[non_exhaustive]
    NoFurtherLikelihoodImprovement {
        /// Criterion the optimizer reported.
        criterion: PrecisionCriterion,
    },
    /// The automatic transform choice was skipped because an observation is
    /// zero or negative, and no transform was used.
    #[non_exhaustive]
    AutomaticTransformNonPositiveInput {
        /// First zero or negative observation in the original input.
        index: usize,
    },
    /// A seasonal filter was replaced because the history is too short.
    #[non_exhaustive]
    SeasonalFilterReplaced {
        /// Requested filter, replaced in one or more calendar periods.
        requested: x11::SeasonalFilter,
        /// Filter used by the adjustment.
        replacement: x11::SeasonalFilter,
    },
    /// ARMA covariance is singular; fitted coefficients remain usable.
    SingularArmaCovariance,
    /// Multiplicative X-11 replaced one or more nonpositive trend estimates.
    #[non_exhaustive]
    NonPositiveTrendReplaced {
        /// Trend-estimation stage at which replacement occurred.
        stage: x11::TrendStage,
        /// Number of estimates replaced during this stage.
        replacements: usize,
    },
    /// Estimation converged but took more iterations than the
    /// reporting threshold of X-13ARIMA-SEATS.
    #[non_exhaustive]
    EstimationRequiredManyIterations {
        /// Cumulative number of estimation iterations.
        iterations: usize,
    },
    /// Automatic model selection dropped the constant without testing it,
    /// because the fit run for the test did not converge.
    #[non_exhaustive]
    ModelSelectionConstantOmitted {
        /// Nonlinear iterations of the fit that did not converge, which the
        /// note X-13ARIMA-SEATS prints leaves out.
        iterations: usize,
    },
    /// The deviance rose during iterative estimation, which continued.
    DevianceIncreased,
    /// A forecast is zero or negative, which multiplicative, log-additive and
    /// pseudo-additive X-11 cannot use, so X-11 adjusted the observed series
    /// without forecasts or backcasts, as X-13ARIMA-SEATS does. The forecasts
    /// and backcasts are still returned.
    ForecastsNotUsed,
    /// The forecasts are usable but a backcast is zero or negative, with the
    /// same result as [`ForecastsNotUsed`](Warning::ForecastsNotUsed).
    BackcastsNotUsed,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SeatsBackcastsIgnored { periods } => write!(
                formatter,
                "SEATS ignores the {periods} requested backcasts, as X-13ARIMA-SEATS does"
            ),
            Self::ModelSpanBackcastsIgnored { periods } => write!(
                formatter,
                "the {periods} requested backcasts were cancelled because the model span starts \
                 after the first observation"
            ),
            Self::ManualOutlierOutsideModelSpan {
                kind: OutlierType::LevelShift,
                period,
            } => write!(
                formatter,
                "the manual level shift at {period} was dropped because the model span must \
                 contain it and an earlier observation"
            ),
            Self::ManualOutlierOutsideModelSpan { kind, period } => write!(
                formatter,
                "the manual {kind} at {period} was dropped because it is outside the model span"
            ),
            Self::ResidualSpectralPeaks { kind } => {
                write!(formatter, "the regARIMA residual spectrum has {kind} peaks")
            }
            Self::X11SpectralPeaks { kind } => {
                write!(formatter, "the X-11 spectra have {kind} peaks")
            }
            Self::X11NoOriginalSeasonalSpectralPeaks => {
                formatter.write_str("the original-series spectrum has no seasonal peak")
            }
            Self::SeasonalityTestUnavailable { test } => {
                write!(formatter, "the {test} is unavailable")
            }
            Self::PseudoAdditiveWithoutForecasts => formatter.write_str(
                "no regARIMA forecasts were requested for pseudo-additive seasonal-difference \
                 forecasting; the observed components remain available",
            ),
            Self::ConstantSeries => {
                formatter.write_str("the series is constant, so preprocessing was skipped")
            }
            Self::ImproperLikelihoodInput => {
                formatter.write_str("the likelihood minimizer received improper input")
            }
            Self::NoFurtherLikelihoodImprovement { criterion } => write!(
                formatter,
                "no further likelihood improvement was possible ({criterion} criterion)"
            ),
            Self::AutomaticTransformNonPositiveInput { index } => write!(
                formatter,
                "automatic transformation selection chose no transform because the observation \
                 at index {index} is not positive"
            ),
            Self::SeasonalFilterReplaced {
                requested,
                replacement,
            } => write!(
                formatter,
                "the {requested} seasonal filter was replaced by the {replacement} filter because \
                 the history is too short"
            ),
            Self::SingularArmaCovariance => formatter.write_str(
                "the ARMA parameter covariance is singular; the fitted coefficients remain usable",
            ),
            Self::NonPositiveTrendReplaced {
                stage,
                replacements,
            } => write!(
                formatter,
                "multiplicative X-11 replaced {replacements} nonpositive {stage} estimates"
            ),
            Self::EstimationRequiredManyIterations { iterations } => write!(
                formatter,
                "estimation converged but needed {iterations} iterations"
            ),
            Self::ModelSelectionConstantOmitted { iterations } => write!(
                formatter,
                "an automatic-model constant test was omitted because its fit did not converge \
                 in {iterations} iterations"
            ),
            Self::DevianceIncreased => {
                formatter.write_str("the deviance increased during estimation")
            }
            Self::ForecastsNotUsed => formatter.write_str(
                "X-11 used no forecasts or backcasts because a forecast is zero or negative",
            ),
            Self::BackcastsNotUsed => formatter.write_str(
                "X-11 used no forecasts or backcasts because a backcast is zero or negative",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::error::{NumericalDetail, NumericalFailure};
    use crate::time::TimeAxis;
    use crate::{Period, Series};

    #[test]
    fn quality_reasons_have_readable_messages() {
        use crate::diagnostics::{X11QualitySeries, X11QualityStage, X11QualityUnavailableReason};
        let constant = X11QualityUnavailableReason::ConstantSeries {
            series: vec![
                X11QualitySeries::Irregular,
                X11QualitySeries::Original {
                    outliers_removed: true,
                },
            ],
        };
        assert_eq!(
            constant.to_string(),
            "constant series: irregular, original series without outlier effects"
        );
        let short = X11QualityUnavailableReason::InsufficientHistory {
            stage: X11QualityStage::M3,
            actual: 2,
            minimum: 3,
        };
        assert_eq!(short.to_string(), "M3: 2 values, fewer than the 3 needed");
    }

    #[test]
    fn final_validation_names_the_first_nonfinite_published_value() {
        use crate::Error;
        use crate::error::{AdjustmentOutput as O, Component};
        let values = (0..72)
            .map(|i| 100.0 + f64::from(i % 12) + f64::from((17 * i * i + 37 * i + 11) % 71) / 10.0)
            .collect::<Vec<_>>();
        let model = crate::ArimaModel::new(
            crate::ArimaOrder::new(0, 1, 1).unwrap(),
            crate::ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let spec = crate::Spec::x11_regarima(
            crate::RegArimaSpec::new(model),
            crate::x11::Options::new().with_mode(crate::x11::Mode::Multiplicative),
        );
        let fitted = || {
            crate::adjust(
                Series::new(Period::monthly(2000, 1).unwrap(), &values),
                &spec,
            )
            .unwrap()
        };
        assert_eq!(fitted().validate_finite(), Ok(()));

        let mut adjustment = fitted();
        adjustment.forecasts.as_mut().unwrap().upper_95[3] = f64::INFINITY;
        assert_eq!(
            adjustment.validate_finite(),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::NonFiniteOutput {
                    output: O::ForecastUpper95,
                    index: 3
                }
            )))
        );
        let mut adjustment = fitted();
        let model = adjustment.model.as_mut().unwrap();
        model.information_criteria.as_mut().unwrap().hannan_quinn = f64::NEG_INFINITY;
        assert_eq!(
            adjustment.validate_finite(),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::NonFiniteOutput {
                    output: O::HannanQuinn,
                    index: 0
                }
            )))
        );
        // Components are checked first.
        adjustment.irregular[5] = f64::NAN;
        assert_eq!(
            adjustment.validate_finite(),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::NonFiniteOutput {
                    output: O::Component(Component::Irregular),
                    index: 5
                }
            )))
        );
    }

    #[test]
    fn regression_covariance_expands_to_symmetric_row_major() {
        assert_eq!(
            super::regression_covariance_row_major(&[], 0),
            Vec::<f64>::new()
        );
        assert_eq!(super::regression_covariance_row_major(&[7.0], 1), [7.0]);
        assert_eq!(
            super::regression_covariance_row_major(&[1.0, 2.0, 3.0], 2),
            [1.0, 2.0, 2.0, 3.0]
        );
        assert_eq!(
            super::regression_covariance_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3),
            [1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn regression_standard_errors_survive_a_singular_arma_covariance() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let values = (0..13)
            .map(|i| 100.0 + ((17 * i * i + 37 * i + 11) % 71) as f64)
            .collect::<Vec<_>>();
        let model = crate::ArimaModel::new(
            crate::ArimaOrder::new(0, 1, 1).unwrap(),
            crate::ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let spec = crate::RegArimaSpec::new(model)
            .with_constant(true)
            .resolve_explicit()
            .unwrap();
        let regression =
            crate::regression::model::RegressionModel::from_spec(spec, axis, values.len()).unwrap();
        let mut fit = crate::arima::estimate(
            &values,
            (&values).into(),
            axis,
            spec,
            regression,
            crate::arima::ArmaStart::Default,
        )
        .unwrap();
        let regression_error = fit.regression_standard_errors.as_ref().unwrap()[0];
        assert!(regression_error > 0.0);
        // The singular covariance is set by hand; the estimation tests cover
        // the fit that produces one.
        fit.arma_covariance = super::ArmaCovariance::Singular;
        fit.parameter_standard_errors = None;
        let residual_spectral = crate::engine::regarima_spectral::compute(
            fit.axis,
            fit.transformed.len(),
            &fit.residuals,
            fit.report().status.converged,
            fit.innovation_variance,
        )
        .unwrap();
        let result = super::ModelResult::from_estimated(fit, None, residual_spectral);
        assert!(
            result
                .parameters()
                .iter()
                .all(|parameter| parameter.standard_error().is_none())
        );
        assert_eq!(
            result.regression_coefficients()[0].standard_error(),
            Some(regression_error)
        );
        assert!(result.regression_covariance().is_some());
    }
}
