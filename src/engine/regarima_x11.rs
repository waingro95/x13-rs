//! Explicit-model regARIMA X-11: `x11ari.f` around an `arima.f` fit, outlier
//! search or calendar AIC test. The shared prologue is `X11Preparation`.

use super::FinalX11Options;
use super::automatic_transform::CandidateModelContext;
use super::post_fit::{PostFitContext, PostFitWarnings};
use super::x11_configuration::{X11Preparation, regression_mode_compatible};
use crate::api::Adjustment;
use crate::arima::estimate;
use crate::error::{SpecProblem, UnsupportedConfiguration, UnsupportedDetail};
use crate::outliers::search::search;
use crate::regarima::X11ModeSelection;
use crate::time::TimeAxis;
use crate::{Error, RegArimaSpec, Warning, x11};

pub(crate) fn adjust(
    values: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    requested: RegArimaSpec,
    requested_mode: X11ModeSelection,
    options: x11::Options,
    final_options: FinalX11Options,
) -> Result<Adjustment, Error> {
    let mut preparation = X11Preparation::new(
        values,
        missing,
        axis,
        requested,
        requested_mode,
        options,
        CandidateModelContext::ExplicitModel,
    )?;
    let editor_warnings: Vec<_> = preparation
        .backcast_warning
        .take()
        .into_iter()
        .chain(preparation.input.dropped_outlier_warnings())
        .collect();
    let spec = preparation.resolved.spec;
    let mode = preparation.mode;
    let easter_requested = options.easter_estimation();
    if super::calendar_selection::requested(spec).is_some() && easter_requested {
        return Err(Error::from(UnsupportedDetail::Configuration(
            UnsupportedConfiguration::CalendarSelection,
        )));
    }
    if !preparation.input.manual_outliers.is_empty()
        && !regression_mode_compatible(spec.transform(), mode)
    {
        return Err(Error::from(
            SpecProblem::IncompatibleRegressionAdjustmentMode {
                transform: spec.transform(),
                mode,
            },
        ));
    }
    if easter_requested {
        let warning = preparation.prepared.take_warning();
        preparation.resolved.warnings.extend(warning);
    }
    preparation.ensure_registration_capacity()?;
    let easter = preparation.prepare_easter()?;
    if mode == x11::Mode::PseudoAdditive {
        // editor.f's filter warning precedes the fit's warnings and the
        // pseudo-additive forecast warning, so it is taken out of the X-11
        // options here.
        let warning = preparation.prepared.take_warning();
        preparation.resolved.warnings.extend(warning);
        if preparation.without_forecasts() {
            preparation
                .resolved
                .warnings
                .push(Warning::PseudoAdditiveWithoutForecasts);
        }
    }

    // x11ari.f bypasses estimation on `IF(Same)` for every model, with or
    // without legacy Easter; the automatic path shares this finish.
    if super::missing::skips_model(values, missing) {
        let mut warnings = editor_warnings;
        warnings.push(Warning::ConstantSeries);
        warnings.extend(preparation.prepared.take_warning());
        warnings.append(&mut preparation.resolved.warnings);
        return preparation.finish_same(easter, final_options.holiday_adjustment, warnings);
    }

    let regression = preparation.regression()?;
    let (estimation_values, fit_context) = preparation.response(&regression, easter.as_ref())?;
    let input = &preparation.input;
    let estimated = if let Some(calendar) = super::calendar_selection::requested(spec) {
        super::calendar_selection::estimate_selected(
            input.values,
            &input.missing,
            input.axis,
            spec,
            calendar,
            regression,
        )?
    } else if spec.outliers().is_some() {
        search(
            &estimation_values,
            fit_context,
            input.axis,
            spec,
            regression,
            preparation.resolved.start,
        )?
        .fit
    } else {
        estimate(
            &estimation_values,
            fit_context,
            input.axis,
            spec,
            regression,
            preparation.resolved.start,
        )?
    };
    let mut warnings = preparation.resolved.warnings;
    warnings.splice(0..0, editor_warnings);
    super::post_fit::finish(
        values,
        estimated,
        PostFitContext {
            series_axis: axis,
            easter,
            mode,
            final_options,
            prepared_options: preparation.prepared,
            transform_selection: preparation.resolved.selection,
            restore_missing_prior: true,
            warnings: PostFitWarnings::BeforeFit(warnings),
            residual_override: None,
        },
    )
}
