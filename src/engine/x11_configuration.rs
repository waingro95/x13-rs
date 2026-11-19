//! The preparation the two regARIMA X-11 pipelines share.
//!
//! `regarima_x11` (an explicit model) and `automatic_model_x11` (automdl) run
//! the same editor.f/x11ari.f prologue: the positive-domain precheck,
//! modelspan narrowing and backcast cancellation, X-11 options on the
//! requested augmented span, the length checks, `trnaic.f` transform
//! resolution and the X-11 mode. After that they share adrgef's capacity
//! stop, the legacy-Easter prepass, the Same bypass, regvar's regression
//! build and the prior-adjusted response. [`X11Preparation`] runs those steps
//! in Build 62's order; it generates no forecasts or backcasts. Each pipeline
//! keeps what differs: its own precondition checks, the order of its editor
//! and candidate warnings, and its estimator (explicit fit, outlier search or
//! calendar AIC test, against automdl).
//!
//! The automatic pipeline also runs the manual-outlier steps (the capacity
//! count and the regression columns). `validate_model_policy` leaves it no
//! manual outliers, so both steps are no-ops there.
//!
//! [`resolve_mode`], [`regression_mode_compatible`] and
//! [`validate_legacy_easter`] are the X-11 rules that request validation,
//! the pipelines and the post-fit restoration all apply.

use super::automatic_transform::{
    CandidateModelContext, OriginalSeries, ResolvedTransform, TransformRequest, remove_fixed_length,
};
use super::model_span::EstimationInput;
use crate::api::Adjustment;
use crate::arima::FitContext;
use crate::error::SpecProblem;
use crate::regarima::{ForecastSpec, X11ModeSelection};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::x11::X11EasterConflict;
use crate::{Error, Frequency, RegArimaSpec, Transform, Warning, x11};

#[derive(Debug, Clone, Copy)]
pub(crate) struct FinalX11Options {
    pub(crate) final_outliers: crate::x11::FinalOutliers,
    pub(crate) temporary_change_allocation: crate::x11::TemporaryChangeAllocation,
    pub(crate) holiday_adjustment: crate::x11::HolidayAdjustment,
}

impl FinalX11Options {
    pub(crate) const fn new(
        final_outliers: crate::x11::FinalOutliers,
        temporary_change_allocation: crate::x11::TemporaryChangeAllocation,
        holiday_adjustment: crate::x11::HolidayAdjustment,
    ) -> Self {
        Self {
            final_outliers,
            temporary_change_allocation,
            holiday_adjustment,
        }
    }
}

/// The steps of regARIMA X-11 that precede estimation, shared by the
/// explicit-model and automatic-model pipelines.
pub(super) struct X11Preparation<'a> {
    /// The full input series, as `editor.f` and `trnaic.f` read it.
    pub(super) original: OriginalSeries<'a>,
    /// The modelspan response, its gaps and its retained manual outliers.
    pub(super) input: EstimationInput<'a>,
    /// The request after editor.f's late-modelspan backcast cancellation.
    pub(super) requested: RegArimaSpec,
    pub(super) backcast_warning: Option<Warning>,
    pub(super) options: x11::Options,
    pub(super) prepared: x11::PreparedOptions,
    pub(super) resolved: ResolvedTransform,
    pub(super) mode: x11::Mode,
}

impl<'a> X11Preparation<'a> {
    pub(super) fn new(
        values: &'a [f64],
        missing: &[usize],
        axis: TimeAxis,
        requested: RegArimaSpec,
        requested_mode: X11ModeSelection,
        options: x11::Options,
        context: CandidateModelContext,
    ) -> Result<Self, Error> {
        if matches!(
            requested_mode,
            X11ModeSelection::Explicit(x11::Mode::Multiplicative | x11::Mode::LogAdditive)
        ) && let Some(index) = crate::numerics::first_nonpositive(values)
        {
            return Err(Error::NonPositiveValue { index });
        }
        let input = EstimationInput::new(values, missing, axis, &requested)?;
        let (requested, backcast_warning) = input.cancel_backcasts(requested)?;
        // editor.f resolves X-11 controls on the requested augmented span, before
        // extend.f may later reject nonpositive forecasts/backcasts.
        let prepared = prepare_options(axis, values.len(), requested.forecast(), options)?;
        crate::api::validate_regarima_x11_length(axis, values.len(), requested.forecast())?;
        // x11ari.f selects the transform before its `IF(Same)` bypass.
        let original = OriginalSeries { values, axis };
        let resolved = TransformRequest {
            context,
            decomposition: options.decomposition(),
            original,
        }
        .resolve(input.values, &input.missing, input.axis, requested.clone())?;
        let mode = resolve_mode(requested_mode, resolved.spec.transform());
        Ok(Self {
            original,
            input,
            requested,
            backcast_warning,
            options,
            prepared,
            resolved,
            mode,
        })
    }

    /// Whether the request generates no forecasts.
    pub(super) fn without_forecasts(&self) -> bool {
        self.requested
            .forecast()
            .forecast_periods(self.original.axis.frequency())
            == 0
    }

    /// gtinpt.f registers the spec's regressors and editor.f a missing-value
    /// regressor per gap while reading the input, so adrgef's capacity stop
    /// precedes x11ari.f's legacy-Easter prepass and Same bypass. Their values
    /// wait for regvar, after both, in [`Self::regression`].
    pub(super) fn ensure_registration_capacity(&self) -> Result<(), Error> {
        RegressionModel::ensure_registration_capacity(
            self.resolved.spec,
            [
                self.input.manual_outliers.len(),
                self.requested.user_regressors().len(),
                self.input.missing.len(),
            ],
        )
    }

    /// x11ari.f's legacy-Easter prepass on the raw series, when requested.
    pub(super) fn prepare_easter(&self) -> Result<Option<x11::PreparedEaster>, Error> {
        if !self.options.easter_estimation() {
            return Ok(None);
        }
        let OriginalSeries { values, axis } = self.original;
        let extensions = self.requested.forecast();
        // xrgdrv suppresses the supported log flow-TD fixed prior (Picktd &&
        // Priadj>1); fitted columns are not present in this raw prepass.
        x11::prepare_easter(
            values,
            axis,
            self.mode,
            &self.prepared,
            usize::from(extensions.forecast_periods(axis.frequency())),
            usize::from(extensions.backcast_periods()),
        )
        .map(Some)
    }

    /// x11ari.f `IF(Same)`: a constant series skips regARIMA estimation.
    pub(super) fn finish_same(
        self,
        easter: Option<x11::PreparedEaster>,
        holiday: x11::HolidayAdjustment,
        warnings: Vec<Warning>,
    ) -> Result<Adjustment, Error> {
        let OriginalSeries { values, axis } = self.original;
        let mut adjustment = super::post_fit::finish_bypass(
            values,
            axis,
            self.mode,
            self.prepared,
            easter,
            holiday,
            warnings,
        )?;
        adjustment.set_bypass_transform_selection(self.resolved.selection);
        Ok(adjustment)
    }

    /// regvar on the modelspan: the spec's columns, then the manual outliers,
    /// user regressors and one missing-value column per gap.
    pub(super) fn regression(&self) -> Result<RegressionModel, Error> {
        let spec = self.resolved.spec;
        let input = &self.input;
        let calendar_enabled = spec.calendar().is_some_and(|calendar| !calendar.is_empty());
        let mut regression = RegressionModel::from_spec(spec, input.axis, input.values.len())?;
        input.add_manual_outliers(&mut regression, spec)?;
        regression.add_user_regressors(
            self.requested.user_regressors(),
            input.axis,
            input.values.len(),
        )?;
        regression.add_missing(&input.missing, input.axis, input.values.len())?;
        if calendar_enabled && !regression_mode_compatible(spec.transform(), self.mode) {
            return Err(Error::from(
                SpecProblem::IncompatibleRegressionAdjustmentMode {
                    transform: spec.transform(),
                    mode: self.mode,
                },
            ));
        }
        Ok(regression)
    }

    /// The estimation response, with the fixed length and any legacy-Easter
    /// prior removed and x11pt1.f's placeholder reset at each gap.
    pub(super) fn response<'e>(
        &self,
        regression: &RegressionModel,
        easter: Option<&'e x11::PreparedEaster>,
    ) -> Result<(Vec<f64>, FitContext<'e>), Error>
    where
        'a: 'e,
    {
        let input = &self.input;
        let mut estimation_values = remove_fixed_length(input.values, regression.fixed_length());
        let fit_context = if let Some(easter) = easter {
            let prior =
                input.remove_easter_prior(&mut estimation_values, easter.observed_factors())?;
            FitContext::with_legacy_easter(input.values, prior)?
        } else {
            FitContext::unadjusted(input.values)
                .with_additive_adjustment(self.mode == x11::Mode::Additive)
        };
        super::missing::reset(&mut estimation_values, &input.missing);
        Ok((estimation_values, fit_context))
    }
}

pub(super) fn prepare_options(
    axis: TimeAxis,
    observed: usize,
    extensions: ForecastSpec,
    options: x11::Options,
) -> Result<x11::PreparedOptions, Error> {
    let (start, requested_count) = extensions.extended_span(axis, observed)?;
    x11::PreparedOptions::with_observed(options, start, requested_count, observed)
}

pub(crate) fn resolve_mode(requested: X11ModeSelection, transform: Transform) -> x11::Mode {
    match requested {
        X11ModeSelection::Explicit(mode) => mode,
        X11ModeSelection::FromTransform => match transform {
            Transform::None => x11::Mode::Additive,
            Transform::Log => x11::Mode::Multiplicative,
        },
    }
}

/// Whether regARIMA effects can be removed in this X-11 mode: the identity
/// transform with additive, the log with any other mode. arima.f tests only
/// Muladd, which pseudo-additive shares with multiplicative.
pub(super) fn regression_mode_compatible(transform: Transform, mode: x11::Mode) -> bool {
    matches!(
        (transform, mode),
        (Transform::None, x11::Mode::Additive)
            | (
                Transform::Log,
                x11::Mode::Multiplicative | x11::Mode::LogAdditive | x11::Mode::PseudoAdditive
            )
    )
}

/// The preconditions of legacy Easter (`x11easter`), shared by direct and
/// regARIMA X-11, in check order: frequency, mode, a fitted regARIMA Easter
/// (never present in direct X-11), then pseudo-additive holiday retention.
pub(crate) fn validate_legacy_easter(
    frequency: Frequency,
    mode: x11::Mode,
    fitted_easter: bool,
    holiday: x11::HolidayAdjustment,
) -> Result<(), Error> {
    let reason = if frequency != Frequency::Monthly {
        X11EasterConflict::Frequency { actual: frequency }
    } else if matches!(mode, x11::Mode::Additive | x11::Mode::LogAdditive) {
        X11EasterConflict::Mode { actual: mode }
    } else if fitted_easter {
        X11EasterConflict::FittedEaster
    } else if mode == x11::Mode::PseudoAdditive && holiday != x11::HolidayAdjustment::Retain {
        X11EasterConflict::PseudoRequiresHolidayRetention
    } else {
        return Ok(());
    };
    Err(Error::from(SpecProblem::IncompatibleX11Easter { reason }))
}
