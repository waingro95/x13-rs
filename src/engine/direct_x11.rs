//! Direct X-11: the `x11` spec without a regARIMA model. The input and
//! options are already validated; this runs X-11, with the legacy-Easter
//! passes when requested, and then the diagnostics.

use super::post_fit::{complete_unmodeled, legacy_calendar};
use crate::api::{Adjustment, AdjustmentOutput, Component};
use crate::error::NumericalDetail;
use crate::time::TimeAxis;
use crate::{Error, Warning, x11};

pub(crate) fn adjust(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    options: x11::Options,
    holiday: x11::HolidayAdjustment,
) -> Result<Adjustment, Error> {
    let easter_requested = options.easter_estimation();
    let mut warnings = Vec::new();
    let (components, easter) = if easter_requested || mode == x11::Mode::PseudoAdditive {
        let mut prepared = x11::PreparedOptions::new(options, axis, values.len())?;
        // editor.f emits filter replacement before the pseudo forecast warning.
        warnings.extend(prepared.take_warning());
        if mode == x11::Mode::PseudoAdditive {
            warnings.push(Warning::PseudoAdditiveWithoutForecasts);
        }
        if easter_requested {
            let (components, easter) = x11::adjust_easter(values, axis, mode, prepared)?;
            (components, Some(easter))
        } else {
            let components = x11::adjust_extended(values, axis, 0..values.len(), mode, prepared)?;
            (components, None)
        }
    } else {
        (x11::adjust(values, axis, mode, options)?, None)
    };
    // A nonfinite component is reported here, before the diagnostics read it.
    validate_component(&components.seasonal, Component::Seasonal)?;
    validate_component(
        &components.seasonally_adjusted,
        Component::SeasonallyAdjusted,
    )?;
    validate_component(&components.trend_cycle, Component::TrendCycle)?;
    validate_component(&components.irregular, Component::Irregular)?;
    let calendar = easter
        .as_ref()
        .map(|easter| legacy_calendar(easter.factors().values(), holiday));
    let completed = complete_unmodeled(components, values, axis, calendar)?;
    let mut adjustment = Adjustment::from_x11_components(axis.start(), completed, warnings);
    if let Some(easter) = easter {
        adjustment.set_x11_easter(easter);
    }
    Ok(adjustment)
}

fn validate_component(values: &[f64], component: Component) -> Result<(), Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(Error::from(NumericalDetail::NonFiniteOutput {
            output: AdjustmentOutput::Component(component),
            index,
        }));
    }
    Ok(())
}
